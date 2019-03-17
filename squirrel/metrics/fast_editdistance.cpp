#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using namespace ::std;

vector<vector<uint32_t>> edit_distance2_with_dp(vector<uint32_t> &x, vector<uint32_t> &y)
{
    uint32_t lx = x.size();
    uint32_t ly = y.size();
    vector<vector<uint32_t>> d(lx + 1, vector<uint32_t>(ly + 1));
    for (uint32_t i = 0; i < lx + 1; i++)
    {
        d[i][0] = i;
    }
    for (uint32_t j = 0; j < ly + 1; j++)
    {
        d[0][j] = j;
    }
    for (uint32_t i = 1; i < lx + 1; i++)
    {
        for (uint32_t j = 1; j < ly + 1; j++)
        {
            d[i][j] = min(min(d[i - 1][j], d[i][j - 1]) + 1,
                          d[i - 1][j - 1] + 2 * (x[i - 1] == y[j - 1] ? 0 : 1));
        }
    }
    return d;
}

vector<vector<uint32_t>> edit_distance2_backtracking(
    vector<vector<uint32_t>> &d,
    vector<uint32_t> &x,
    vector<uint32_t> &y,
    uint32_t terminal_symbol)
{
    vector<uint32_t> seq;
    vector<vector<uint32_t>> edit_seqs(x.size() + 2, vector<uint32_t>());
    /* 
    edit_seqs: 
    0~x.size() cell is the insertion sequences
    last cell is the delete sequence
    */

    if (x.size() == 0)
    {
        edit_seqs[0] = y;
        return edit_seqs;
    }

    uint32_t i = d.size() - 1;
    uint32_t j = d[0].size() - 1;

    while ((i >= 0) && (j >= 0))
    {

        if ((i == 0) && (j == 0))
            break;

        if ((j > 0) && (d[i][j - 1] < d[i][j]))
        {
            seq.push_back(1); // insert
            seq.push_back(y[j - 1]);
            j--;
        }
        else if ((i > 0) && (d[i - 1][j] < d[i][j]))
        {
            seq.push_back(2); // delete
            seq.push_back(x[i - 1]);
            i--;
        }
        else
        {
            seq.push_back(3); // keep
            seq.push_back(x[i - 1]);
            i--;
            j--;
        }
    }

    uint32_t prev_op, op, s, word;
    prev_op = 0, s = 0;
    for (uint32_t i = 0; i < seq.size() / 2; i++)
    {
        op = seq[seq.size() - 2 * i - 2];
        word = seq[seq.size() - 2 * i - 1];
        if (prev_op != 1)
        {
            s++;
        }
        if (op == 1) // insert
        {
            edit_seqs[s - 1].push_back(word);
        }
        else if (op == 2) // delete
        {
            edit_seqs[x.size() + 1].push_back(1);
        }
        else
        {
            edit_seqs[x.size() + 1].push_back(0);
        }

        prev_op = op;
    }

    for (uint32_t k = 0; k < edit_seqs.size(); k++)
    {
        if (edit_seqs[k].size() == 0)
        {
            edit_seqs[k].push_back(terminal_symbol);
        }
    }
    return edit_seqs;
}

vector<uint32_t> compute_ed2(vector<vector<uint32_t>> &xs, vector<vector<uint32_t>> &ys)
{
    vector<uint32_t> distances(xs.size());
    for (uint32_t i = 0; i < xs.size(); i++)
    {
        vector<vector<uint32_t>> d = edit_distance2_with_dp(xs[i], ys[i]);
        distances[i] = d[xs[i].size()][ys[i].size()];
    }
    return distances;
}

vector<vector<vector<uint32_t>>> suggested_ed2_path(
    vector<vector<uint32_t>> &xs,
    vector<vector<uint32_t>> &ys,
    uint32_t terminal_symbol)
{
    vector<vector<vector<uint32_t>>> seq(xs.size());
    for (uint32_t i = 0; i < xs.size(); i++)
    {
        vector<vector<uint32_t>> d = edit_distance2_with_dp(xs[i], ys[i]);
        seq[i] = edit_distance2_backtracking(d, xs[i], ys[i], terminal_symbol);
    }
    return seq;
}

namespace py = pybind11;
PYBIND11_PLUGIN(fast_editdistance)
{
    py::module m("fast_editdistance", "Edit distance made by pybind11");
    m.def("compute_ed2", &compute_ed2);
    m.def("suggested_ed2_path", &suggested_ed2_path);
    return m.ptr();
}