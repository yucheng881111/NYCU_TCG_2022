#pragma once
#include <bits/stdc++.h>
#include <torch/torch.h>
#include "board.h"

class MovingStates {
public:
	std::deque<board> states;

	MovingStates() {
		states.push_back(board());
		states.push_back(board());
		states.push_back(board());
	}

	void add_state(board b) {
		states.pop_front();
		states.push_back(b);
	}

	void clean() {
		states.clear();
		states.push_back(board());
		states.push_back(board());
		states.push_back(board());
	}

};

torch::Tensor getTensor(board a, board b, board c) {
	auto tmp_data = torch::zeros({1, 7, 9, 9}); // N, C, H, W
	a.rotate_left();
	b.rotate_left();
	c.rotate_left();

	// current board
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			if (c[i][j] == board::black) {
				tmp_data.index_put_({0, 0, i, j}, 1);
			} else if (c[i][j] == board::white) {
				tmp_data.index_put_({0, 3, i, j}, -1);
			}
		}
	}

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			if (b[i][j] == board::black) {
				tmp_data.index_put_({0, 1, i, j}, 1);
			} else if (b[i][j] == board::white) {
				tmp_data.index_put_({0, 4, i, j}, -1);
			}
		}
	}

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			if (a[i][j] == board::black) {
				tmp_data.index_put_({0, 2, i, j}, 1);
			} else if (a[i][j] == board::white) {
				tmp_data.index_put_({0, 5, i, j}, -1);
			}
		}
	}

	if (c.info().who_take_turns == board::black) {
		for (int i = 0; i < 9; ++i) {
			for (int j = 0; j < 9; ++j) {
				tmp_data.index_put_({0, 6, i, j}, 1);
			}
		}
	} else {
		for (int i = 0; i < 9; ++i) {
			for (int j = 0; j < 9; ++j) {
				tmp_data.index_put_({0, 6, i, j}, -1);
			}
		}
	}

	return tmp_data;
}


