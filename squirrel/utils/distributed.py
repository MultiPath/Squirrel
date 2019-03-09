import pickle

import torch
import torch.distributed as dist


# ===== util fucntions ===== #
def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def gather_tensor(tensor, world_size=1):
    tensor_list = [tensor.clone() for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)

    return rt / dist.get_world_size()


def reduce_dict(info_dict):
    for it in info_dict:
        p = info_dict[it].clone()
        dist.all_reduce(p, op=dist.reduce_op.SUM)
        info_dict[it] = p / dist.get_world_size()


def all_gather_list(data, max_size=32768):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size) for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 3 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 3))

    assert max_size < 255 * 255 * 256
    in_buffer[0] = enc_size // (255 * 255
                                )  # this encoding works for max_size < 16M
    in_buffer[1] = (enc_size % (255 * 255)) // 255
    in_buffer[2] = (enc_size % (255 * 255)) % 255
    in_buffer[3:enc_size + 3] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    result = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * 255 * item(out_buffer[0])) + 255 * item(
            out_buffer[1]) + item(out_buffer[2])
        result.append(pickle.loads(bytes(out_buffer[3:size + 3].tolist())))
    return result


def gather_dict(info_dict, max_size=2**20):
    for w in info_dict:
        new_v = []

        try:
            results = all_gather_list(info_dict[w], max_size)
        except ValueError:
            results = all_gather_list(info_dict[w], max_size * 2)

        for v in results:
            if isinstance(v, list):
                new_v += v
            else:
                new_v.append(v)

        info_dict[w] = new_v
