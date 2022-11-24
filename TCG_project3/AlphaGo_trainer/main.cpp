#include <bits/stdc++.h>
#include <torch/torch.h>
#include "neural/network.h"

//using namespace std;

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
	srand((unsigned)time(NULL));

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

auto Sample_Batch_Eval(int batch_size) {
	srand((unsigned)time(NULL));
	
	auto tmp_data = torch::zeros({batch_size, channel_size, 9, 9}); // N, C, H, W
	auto tmp_p_label = torch::zeros({batch_size, 81});
	auto tmp_v_label = torch::zeros({batch_size});
	int len = replay_buffer_eval.size();

	for (int i = 0; i < batch_size; ++i) {
		
		int data_idx = rand() % len;
		int trajectory_idx;
		int trajectory_len = replay_buffer_eval[data_idx].size();
		

		while (1) {
			trajectory_idx = rand() % trajectory_len;
			if ((trajectory_idx - (seq_len - 1) >= 0) && (trajectory_idx != trajectory_len - 1)) {
				break;
			}
		}

		int k = 0;
		for (int j = trajectory_idx; j >= trajectory_idx - (seq_len - 1); --j) {
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					if (replay_buffer_eval[data_idx][j][m][n] == 1) {
						tmp_data.index_put_({i, k, m, n}, 1);
					} else if (replay_buffer_eval[data_idx][j][m][n] == -1) {
						tmp_data.index_put_({i, k + seq_len, m, n}, -1);
					}
				}
			}
			k++;
		}
		
		int p = replay_buffer_eval_policy[data_idx][trajectory_idx];   // policy target
		int v = is_win_eval[data_idx];                                 // value target

		if (p < 81) {  // current play black
			// last channel (play black => all 1)
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					tmp_data.index_put_({i, channel_size - 1, m, n}, 1);
				}
			}
			tmp_p_label.index_put_({i, p}, 1);
			if (v == 1) {
				tmp_v_label.index_put_({i}, 1);   // now play black and black win
			} else {
				tmp_v_label.index_put_({i}, -1);  // now play black and white win
			}

		} else {       // current play white
			// last channel (play white => all -1)
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					tmp_data.index_put_({i, channel_size - 1, m, n}, -1);
				}
			}
			tmp_p_label.index_put_({i, p - 81}, 1);
			if (v == -1) {
				tmp_v_label.index_put_({i}, 1);   // now play white and white win
			} else {
				tmp_v_label.index_put_({i}, -1);  // now play white and black win
			}
		}
		
	}
		
	//tmp_data = tmp_data.to(device);
	//tmp_p_label = tmp_p_label.to(device);
	//tmp_v_label = tmp_v_label.to(device);
	return std::make_tuple(tmp_data, tmp_p_label, tmp_v_label);

}


auto Sample_Batch(int batch_size) {
	srand((unsigned)time(NULL));

	auto tmp_data = torch::zeros({batch_size, channel_size, 9, 9}); // N, C, H, W
	auto tmp_p_label = torch::zeros({batch_size, 81});
	auto tmp_v_label = torch::zeros({batch_size});
	int len = replay_buffer.size();

	for (int i = 0; i < batch_size; ++i) {
		
		int data_idx = rand() % len;
		int trajectory_idx;
		int trajectory_len = replay_buffer[data_idx].size();
		

		while (1) {
			trajectory_idx = rand() % trajectory_len;
			if ((trajectory_idx - (seq_len - 1) >= 0) && (trajectory_idx != trajectory_len - 1)) {
				break;
			}
		}

		int k = 0;
		for (int j = trajectory_idx; j >= trajectory_idx - (seq_len - 1); --j) {
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					if (replay_buffer[data_idx][j][m][n] == 1) {
						tmp_data.index_put_({i, k, m, n}, 1);
					} else if (replay_buffer[data_idx][j][m][n] == -1) {
						tmp_data.index_put_({i, k + seq_len, m, n}, -1);
					}
				}
			}
			k++;
		}
		
		int p = replay_buffer_policy[data_idx][trajectory_idx];   // policy target
		int v = is_win[data_idx];                                 // value target

		if (p < 81) {  // current play black
			// last channel (play black => all 1)
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					tmp_data.index_put_({i, channel_size - 1, m, n}, 1);
				}
			}
			tmp_p_label.index_put_({i, p}, 1);
			if (v == 1) {
				tmp_v_label.index_put_({i}, 1);   // now play black and black win
			} else {
				tmp_v_label.index_put_({i}, -1);  // now play black and white win
			}

		} else {       // current play white
			// last channel (play white => all -1)
			for (int m = 0; m < 9; ++m) {
				for (int n = 0; n < 9; ++n) {
					tmp_data.index_put_({i, channel_size - 1, m, n}, -1);
				}
			}
			tmp_p_label.index_put_({i, p - 81}, 1);
			if (v == -1) {
				tmp_v_label.index_put_({i}, 1);   // now play white and white win
			} else {
				tmp_v_label.index_put_({i}, -1);  // now play white and black win
			}
		}
		
	}
		
	//tmp_data = tmp_data.to(device);
	//tmp_p_label = tmp_p_label.to(device);
	//tmp_v_label = tmp_v_label.to(device);
	return std::make_tuple(tmp_data, tmp_p_label, tmp_v_label);
}

int main(){
	
	srand((unsigned)time(NULL));
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	// planes, height, width, filters, num_res_blocks, policy_size
	const auto net_op = az::NetworkOptions{channel_size, 9, 9, 128, 2, 81};
	
	// use one network with two head
	//az::AlphaZeroNetwork net(net_op);
	//net->to(device);

	// use two seperate network
	az::AlphaZeroNetwork policy_net(net_op);
	policy_net->to(device);
	az::AlphaZeroNetwork value_net(net_op);
	value_net->to(device);

	open_history();

	num_of_testing_data = replay_buffer_eval.size();

	int replay_buffer_size = replay_buffer.size();
	int replay_buffer_eval_size = replay_buffer_eval.size();
	std::cout << "replay buffer size: " << replay_buffer_size << std::endl;
	std::cout << "replay buffer eval size: " << replay_buffer_eval_size << std::endl;

	// debug
	//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp = Sample_Batch(batch_size);
	//auto data = std::get<0>(tp).to(device);
	//auto p_label = std::get<1>(tp).to(device);
	//auto v_label = std::get<2>(tp).to(device);
	//std::cout << data << std::endl;
	//std::cout << p_label << std::endl;
	//std::cout << v_label << std::endl;
	
	int epoch = 150;
	//torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.02).weight_decay(1e-4));

	torch::optim::Adam optimizer_p(policy_net->parameters(), torch::optim::AdamOptions(0.02).weight_decay(1e-4));
	torch::optim::Adam optimizer_v(value_net->parameters(), torch::optim::AdamOptions(0.02).weight_decay(1e-4));

	for(int i = 0; i < epoch; ++i) {
		//net->train();
		policy_net->train();
		value_net->train();
		float batch_total_loss_v = 0.0;
		float batch_total_loss_p = 0.0;
		int cnt_v = 0, cnt_p = 0;
		for (int k = 0; k < num_of_training_data/10; ++k) {
			//optimizer.zero_grad();
			optimizer_p.zero_grad();
			optimizer_v.zero_grad();
			std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp = Sample_Batch(batch_size);

			auto data = std::get<0>(tp).to(device);
			auto p_label = std::get<1>(tp).to(device);
			auto v_label = std::get<2>(tp).to(device);
      // policy update
			auto [v_skip, p_out] = policy_net->forward(data);
			auto p_loss = torch::mean(-torch::sum(p_label * torch::log_softmax(p_out, 1), 1));
			batch_total_loss_p += p_loss.template item<float>();
			p_loss.backward();
			optimizer_p.step();
      
      optimizer_p.zero_grad();
      optimizer_v.zero_grad();

      // value update
      auto [v_out, p_skip] = value_net->forward(data);
      auto v_loss = torch::mse_loss(v_out.view(-1), v_label);
			v_loss.backward();
			optimizer_v.step();

      /*
      // use one module
      auto [v_out, p_out] = net->forward(data);
      auto v_loss = torch::mse_loss(v_out.view(-1), v_label);
      auto p_loss = torch::mean(-torch::sum(p_label * torch::log_softmax(p_out, 1), 1));
      batch_total_loss_v += v_loss.template item<float>();
      batch_total_loss_p += p_loss.template item<float>();
      auto total_loss = v_loss + p_loss;
      total_loss.backward();
      optimizer.step();
      */

			// calculate accuracy
			for (int j = 0; j < batch_size; ++j) {
				float v_truth = v_label[j].template item<float>();
				float v_pred = v_out.view(-1)[j].template item<float>();
				if (v_truth == 1 && v_pred >= 0) {
					cnt_v++;
				} else if (v_truth == -1 && v_pred < 0) {
					cnt_v++;
				}

				auto p_truth = std::get<1>(torch::max(p_label, 1))[j];
				auto p_pred = std::get<1>(torch::max(p_out, 1))[j];
				float p_truth_f = p_truth.template item<float>();
				float p_pred_f = p_pred.template item<float>();
				
				if (p_truth_f == p_pred_f) {
					cnt_p++;
				}
			}
		}
		float batch_total_loss = batch_total_loss_v + batch_total_loss_p;
		std::cout << "epoch " << i + 1 << ":\n";
		std::cout << "value acc: " << (float)cnt_v / (batch_size*num_of_training_data/10) << "  ";
		std::cout << "policy acc: " << (float)cnt_p / (batch_size*num_of_training_data/10) << std::endl;
		std::cout << "value loss: " << batch_total_loss_v << "  ";
		std::cout << "policy loss: " << batch_total_loss_p << "  ";
		std::cout << "total loss: " << batch_total_loss << std::endl;
		
		
		if ((i + 1) % 5 == 0) {
			std::cout << "eval:" << std::endl;
			//net->eval();
			policy_net->eval();
			value_net->eval();
			int cnt_v = 0, cnt_p = 0;
			float batch_total_loss_v = 0.0;
			float batch_total_loss_p = 0.0;
			for (int k = 0; k < num_of_testing_data/10; ++k) {
				std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tp_eval = Sample_Batch_Eval(batch_size);
				
				auto data_eval = std::get<0>(tp_eval).to(device);
				auto p_label_eval = std::get<1>(tp_eval).to(device);
				auto v_label_eval = std::get<2>(tp_eval).to(device);

				//auto [v_out_eval, p_out_eval] = net->forward(data_eval);
				auto [v_out_eval_skip, p_out_eval] = policy_net->forward(data_eval);
				auto [v_out_eval, p_out_eval_skip] = value_net->forward(data_eval);

				auto v_loss = torch::mse_loss(v_out_eval.view(-1), v_label_eval);
				auto p_loss = torch::mean(-torch::sum(p_label_eval * torch::log_softmax(p_out_eval, 1), 1));
				
				batch_total_loss_v += v_loss.template item<float>();
				batch_total_loss_p += p_loss.template item<float>();
				
				for (int j = 0; j < batch_size; ++j) {
					float v_truth = v_label_eval[j].template item<float>();
					float v_pred = v_out_eval.view(-1)[j].template item<float>();
					if (v_truth == 1 && v_pred >= 0) {
						cnt_v++;
					} else if (v_truth == -1 && v_pred < 0) {
						cnt_v++;
					}

					auto p_truth = std::get<1>(torch::max(p_label_eval, 1))[j];
					auto p_pred = std::get<1>(torch::max(p_out_eval, 1))[j];
					float p_truth_f = p_truth.template item<float>();
					float p_pred_f = p_pred.template item<float>();
					
					if (p_truth_f == p_pred_f) {
						cnt_p++;
					}
					
				}
			}
			
			std::cout << "value acc: " << (float)cnt_v / (batch_size*num_of_testing_data/10) << "  ";
			std::cout << "policy acc: " << (float)cnt_p / (batch_size*num_of_testing_data/10) << std::endl;
			std::cout << "value loss: " << batch_total_loss_v << "  ";
			std::cout << "policy loss: " << batch_total_loss_p << "  ";
			std::cout << "total loss: " << batch_total_loss_v + batch_total_loss_p << std::endl << std::endl;

			torch::save(policy_net, "policy/epoch" + std::to_string(i + 1) + "_weights.pt");
      torch::save(value_net, "value/epoch" + std::to_string(i + 1) + "_weights.pt");
		}
		
	}

return 0;
}







