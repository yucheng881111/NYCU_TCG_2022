#include <bits/stdc++.h>
#include <torch/torch.h>
#include "neural/network.h"

#define EVAL_VALUE

std::vector<std::vector<std::vector<std::vector<int>>>> replay_buffer;
std::vector<std::vector<int>> replay_buffer_policy;
std::vector<int> is_win;

std::vector<std::vector<std::vector<std::vector<int>>>> replay_buffer_eval;
std::vector<std::vector<int>> replay_buffer_eval_policy;
std::vector<int> is_win_eval;

int seq_len = 3;
int channel_size = seq_len * 2 + 1;
int batch_size = 512;
int num_of_training_data = 13000;
int num_of_testing_data = 0;

std::pair<int, int> reconstruct(int pos) {
	int i, j;
	if (pos >= 81) {  // white
		pos -= 81;
		j = pos / 9;
		i = 9 - (pos % 9) - 1;
		
	} else {         // black
		j = pos / 9;
		i = 9 - (pos % 9) - 1;
	}

	return {i, j};
}


void open_history() {

	std::fstream ifs("history.txt", std::ios::in);
	std::fstream ifs2("win_or_loss.txt", std::ios::in);
	
	std::string s;
	int cnt = 0;
	while (std::getline(ifs2, s)) {
        std::stringstream ss(s);
        int i;
        ss >> i;
        
        if (cnt >= num_of_training_data) {
        	is_win_eval.push_back(i); // value target
        } else {
			is_win.push_back(i); // value target
        }
		
        cnt++;
	}

	
	cnt = 0;
    while (std::getline(ifs, s)) {
		std::vector<int> vec;
        std::stringstream ss(s);
		int i;
		while (ss >> i) {
			vec.push_back(i);
		}
		
		int len = vec.size();

		std::vector<std::vector<std::vector<int>>> trajectory;
		std::vector<std::vector<int>> board(9, std::vector<int>(9, 0));
		trajectory.push_back(board);
		for (auto &v : vec) {
			std::pair<int, int> p = reconstruct(v);
			if (v < 81) {     // black
				board[p.first][p.second] = 1;
				trajectory.push_back(board);
			} else {          // white
				board[p.first][p.second] = -1;
				trajectory.push_back(board);
			}

		}
		
		if (cnt >= num_of_training_data) {
			replay_buffer_eval.push_back(trajectory);
			replay_buffer_eval_policy.push_back(vec);
		} else {
			replay_buffer.push_back(trajectory);
			replay_buffer_policy.push_back(vec);
		}
		
		cnt++;
    }

    

    ifs.close();
    ifs2.close();

}

int main() {
	open_history();
	auto tmp_data = torch::zeros({1, channel_size, 9, 9});
	auto tmp_p_label = torch::zeros({1, 81});
	auto tmp_v_label = torch::zeros({1});
	/*
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			tmp.index_put_({0, 6, i, j}, 1);
		}
	}
	*/
  
  int idx = 1;
  std::cout << "data idx: ";
  std::cin >> idx;
	int len = replay_buffer[idx].size();
	std::cout << "data index " << idx << " has size: " << len << std::endl;

	#ifdef EVAL_VALUE
		for (int trajectory_idx = (seq_len-1); trajectory_idx <= len-(seq_len-1); ++trajectory_idx) {
			int k = 0;
			std::cout << "trajectory idx: " << trajectory_idx << std::endl;
			for (int j = trajectory_idx; j >= trajectory_idx - (seq_len - 1); --j) {
				//std::cout << replay_buffer[0][j] << std::endl;
				for (int m = 0; m < 9; ++m) {
					for (int n = 0; n < 9; ++n) {
						if (replay_buffer[idx][j][m][n] == 1) {
							tmp_data.index_put_({0, k, m, n}, 1);
						} else if (replay_buffer[idx][j][m][n] == -1) {
							tmp_data.index_put_({0, k + seq_len, m, n}, -1);
						}
					}
				}
				k++;
			}

			int p = replay_buffer_policy[idx][trajectory_idx];   // policy target
			int v = is_win[idx];                                 // value target

			if (p < 81) {  // current play black
				// last channel (play black => all 1)
				for (int m = 0; m < 9; ++m) {
					for (int n = 0; n < 9; ++n) {
						tmp_data.index_put_({0, channel_size - 1, m, n}, 1);
					}
				}
				
				tmp_p_label.index_put_({0, p}, 1);
				if (v == 1) {
					tmp_v_label.index_put_({0}, 1);   // now play black and black win
				} else {
					tmp_v_label.index_put_({0}, -1);  // now play black and white win
				}
				
			} else {       // current play white
				// last channel (play white => all -1)
				for (int m = 0; m < 9; ++m) {
					for (int n = 0; n < 9; ++n) {
						tmp_data.index_put_({0, channel_size - 1, m, n}, -1);
					}
				}
				
				tmp_p_label.index_put_({0, p - 81}, 1);
				if (v == -1) {
					tmp_v_label.index_put_({0}, 1);   // now play white and white win
				} else {
					tmp_v_label.index_put_({0}, -1);  // now play white and black win
				}
				
			}


			torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
			// planes, height, width, filters, num_res_blocks, policy_size
			const auto net_op = az::NetworkOptions{channel_size, 9, 9, 128, 2, 81};
			az::AlphaZeroNetwork net(net_op);
			torch::load(net, "value/epoch150_weights.pt");
			net->to(device);
			tmp_data = tmp_data.to(device);
			net->eval();
			auto [v_out, p_out] = net->forward(tmp_data);
			std::cout << tmp_v_label << "  ";
			float v_pred = v_out.view(-1)[0].template item<float>();
			std::cout << "value: " << v_pred << std::endl;
		}

	#endif



	int k = 0;
	int trajectory_idx = 2;
	std::cout << "input trajectory index: ";
	std::cin >> trajectory_idx;
	for (int j = trajectory_idx; j >= trajectory_idx - (seq_len - 1); --j) {
		//std::cout << replay_buffer[0][j] << std::endl;
		for (int m = 0; m < 9; ++m) {
			for (int n = 0; n < 9; ++n) {
				if (replay_buffer[0][j][m][n] == 1) {
					tmp_data.index_put_({0, k, m, n}, 1);
				} else if (replay_buffer[0][j][m][n] == -1) {
					tmp_data.index_put_({0, k + seq_len, m, n}, -1);
				}
			}
		}
		k++;
	}

	int p = replay_buffer_policy[0][trajectory_idx];   // policy target
	int v = is_win[0];                                 // value target

	if (p < 81) {  // current play black
		// last channel (play black => all 1)
		for (int m = 0; m < 9; ++m) {
			for (int n = 0; n < 9; ++n) {
				tmp_data.index_put_({0, channel_size - 1, m, n}, 1);
			}
		}
		
		tmp_p_label.index_put_({0, p}, 1);
		if (v == 1) {
			tmp_v_label.index_put_({0}, 1);   // now play black and black win
		} else {
			tmp_v_label.index_put_({0}, -1);  // now play black and white win
		}
		
	} else {       // current play white
		// last channel (play white => all -1)
		for (int m = 0; m < 9; ++m) {
			for (int n = 0; n < 9; ++n) {
				tmp_data.index_put_({0, channel_size - 1, m, n}, -1);
			}
		}
		
		tmp_p_label.index_put_({0, p - 81}, 1);
		if (v == -1) {
			tmp_v_label.index_put_({0}, 1);   // now play white and white win
		} else {
			tmp_v_label.index_put_({0}, -1);  // now play white and black win
		}
		
	}



	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	// planes, height, width, filters, num_res_blocks, policy_size
	const auto net_op = az::NetworkOptions{channel_size, 9, 9, 128, 2, 81};
	az::AlphaZeroNetwork net(net_op);
	torch::load(net, "epoch30_weights.pt");
	net->to(device);
	tmp_data = tmp_data.to(device);
	net->eval();
	auto [v_out, p_out] = net->forward(tmp_data);

	std::cout << tmp_data << std::endl;
	std::cout << tmp_p_label << std::endl;
	std::cout << tmp_v_label << std::endl;

	float v_pred = v_out.view(-1)[0].template item<float>();
	std::cout << "value: " << v_pred << std::endl << std::endl;
	p_out = torch::softmax(p_out, 1);

	for (int j = 0; j < 81; ++j) {
		float prob = p_out[0][j].template item<float>();
		std::cout << "action: " << j << " prob: " << prob << std::endl;
	}

return 0;	
}






