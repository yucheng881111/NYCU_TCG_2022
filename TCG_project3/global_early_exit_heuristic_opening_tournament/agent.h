/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"

#include <bits/stdc++.h>
#include <omp.h>

#define ParallelAverageSelection
//#define ParallelMajorityVote
//#define Normal

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b, int steps) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
		else
			engine.seed((unsigned)time(NULL));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state, int steps) {
		int N = meta["N"];
		float c = meta["c"];
		if (N) {
			#ifdef ParallelAverageSelection
			int thread_num = omp_get_num_procs();
			omp_set_num_threads(thread_num);

			std::vector<std::vector<int>> average_selection(thread_num, std::vector<int>(81, 0));
			std::vector<node*> root_vec;

			for (int i = 0; i < thread_num; ++i) {
				node* root = new node(state);
				root_vec.push_back(root);
			}

			int split = N / 1000;

			#pragma omp parallel
			{
				int id = omp_get_thread_num();
				root_vec[id]->MCTS(N / 4, engine, c);
				/*
				for (auto &all_child : root_vec[id]->child) {
					average_selection[id][all_child.first] = (all_child.second)->total_cnt;
				}
				*/
			}
			
			for (int i = split / 4 + 1; i <= split; ++i) {

				#pragma omp parallel
				{
					int id = omp_get_thread_num();
					root_vec[id]->MCTS(N / split, engine, c);

					for (auto &all_child : root_vec[id]->child) {
						average_selection[id][all_child.first] = (all_child.second)->total_cnt;
					}
					
				}

				// early exit
				std::vector<int> result(81, 0);
				for (int j = 0; j < 81; ++j) {
					for (int i = 0; i < thread_num; ++i) {
						result[j] += average_selection[i][j];
					}
				}

				int max1 = -std::numeric_limits<int>::max();
				int max2 = -std::numeric_limits<int>::max();
				for (auto &r : result) {
					int tmp = r;
					if (tmp > max1) {
						max2 = max1;
						max1 = tmp;
					} else if (tmp > max2) {
						max2 = tmp;
					}
				}

				float f = 0.5;
				if (max2 + f * (N * thread_num - ((i * N / split) * thread_num)) < max1) {
					break;
				}
			}

			for (auto &root : root_vec) {
				delete_tree(root);
			}

			std::vector<int> result(81, 0);
			for (int j = 0; j < 81; ++j) {
				for (int i = 0; i < thread_num; ++i) {
					result[j] += average_selection[i][j];
				}
			}
			
			auto iter = max_element(result.begin(), result.end());
			if((*iter) == 0){
				return action();
			}

			return action::place(iter - result.begin(), state.info().who_take_turns);
			

			#endif

			#ifdef ParallelMajorityVote
			// root parallelizing
			int thread_num = omp_get_num_procs();
			omp_set_num_threads(thread_num);
			std::vector<int> majority_vote(thread_num, 0);

			//std::fstream debug("record.txt", std::ios::app);

			#pragma omp parallel
			{
				int id = omp_get_thread_num();
				node* root = new node(state);
				int vote_move = root->MCTS(N, engine, c);
				delete_tree(root);
				majority_vote[id] = vote_move;
			}

			std::vector<int> vote_result(81, 0);
			for(auto &v : majority_vote){
				if(v != -1){
					//debug << v << " ";
					vote_result[v]++;
				}
			}

			//debug << std::endl;
			//debug.close();

			std::vector<int>::iterator iter = max_element(vote_result.begin(), vote_result.end());
			if((*iter) == 0){
				return action();
			}

			int result = iter - vote_result.begin();
			return action::place(result, state.info().who_take_turns);

			#endif

			#ifdef Normal

			node* root = new node(state);
			int result = root->MCTS(N, engine, c);
			return action::place(result, state.info().who_take_turns);

			#endif
		}

		/*
		// random play
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		*/
		return action();
	}

	class node : board {
	public:
		int win_cnt;
		int total_cnt;
		int place_pos;
		//std::vector<node*> child;
		std::unordered_map<int, node*> child;
		node* parent;

		node(const board& state, int m = -1): board(state), place_pos(m), win_cnt(0), total_cnt(0), parent(nullptr) {}

		float win_rate(){
			if(total_cnt == 0){
				return 0.0;
			}
			
			return (float)win_cnt / total_cnt;
		}

		float ucb(float c){
			if(parent->total_cnt == 0 || total_cnt == 0){
				return win_rate();
			}
			
			return win_rate() + c * std::sqrt(std::log(parent->total_cnt) / total_cnt);
		}

		int MCTS(int N, std::default_random_engine& engine, float ucb_c){
			// 1. select  2. expand  3. simulate  4. back propagate

			// debug
			//std::fstream debug("record.txt", std::ios::app);
			
			for(int i = 0; i < N; ++i){
				// early exit
				/*
				if ((i + 1) % 1000 == 0) {
					int max1 = -std::numeric_limits<int>::max();
					int max2 = -std::numeric_limits<int>::max();
					for (auto &m_child : child) {
						int tmp = m_child.second->total_cnt;
						if (tmp > max1) {
							max2 = max1;
							max1 = tmp;
						} else if (tmp > max2) {
							max2 = tmp;
						}
					}

					if (max2 + (N - i) < max1) {
						break;
					}
				}
				*/
				// select
				//debug << "select" << std::endl;
				std::vector<node*> path = select_root_to_leaf(info().who_take_turns, ucb_c);
				// expand
				//debug << "expand" << std::endl;
				node* leaf = path.back();
				node* expand_node = leaf->expand_from_leaf(engine);
				if(expand_node != leaf){
					path.push_back(expand_node);
				}
				// simulate
				//debug << "simulate" << std::endl;
				unsigned winner = path.back()->simulate_winner(engine);
				// backpropagate
				//debug << "backpropagate" << std::endl;
				back_propagate(path, winner);
			}

			//debug.close();
			return select_action();
		}

		int select_action(){
			// select child node with most visit count
			if(child.size() == 0){
				return -1;
			}

			int max_visit = -std::numeric_limits<int>::max();
			node* c;
			for(auto &ch : child){
				int tmp = ch.second->total_cnt;
				if(tmp > max_visit){
					max_visit = tmp;
					c = ch.second;
				}
			}
			
			return c->place_pos;
		}

		std::vector<node*> select_root_to_leaf(unsigned who, float ucb_c){
			std::vector<node*> vec;
			node* curr = this;
			vec.push_back(curr);
			while(!curr->is_leaf()){
				// select node who has the highest ucb score
				float max_score = -std::numeric_limits<float>::max();
				node* c;
				if(curr->child.size() == 0){
					break;
				}
				for(auto &curr_child : curr->child){
					float tmp = (curr_child.second)->ucb(ucb_c);
					
					if(tmp > max_score){
						max_score = tmp;
						c = curr_child.second;
					}

				}
				vec.push_back(c);
				curr = c;
			}

			return vec;
		}

		bool is_leaf(){
			int cnt = 0;
			for(int i = 0; i < 81; i++){
				if(board(*this).place(i) == board::legal){
					cnt++;
				}
			}
			// check if fully expanded (leaf == not fully expanded)
			return !(cnt > 0 && child.size() == cnt);
		}

		node* expand_from_leaf(std::default_random_engine& engine){
			board b = *this;
			std::vector<int> vec = all_space(engine);
			bool success_placed = 0;
			int pos = -1;

			for(int i = 0; i < vec.size(); ++i){
				b = *this;
				if(b.place(vec[i]) == board::legal && (*this).child.count(vec[i]) == 0){
					pos = vec[i];
					success_placed = 1;
					break;
				}
			}

			if(success_placed){
				node* new_node = new node(b, pos);
				//this->child.push_back(new_node);
				this->child[pos] = new_node;
				new_node->parent = this;
				return new_node;
			}else{
				return this;
			}
		}

		unsigned simulate_winner(std::default_random_engine& engine){
			board b = *this;
			std::vector<int> vec = all_space(engine);
			std::queue<int> q;
			for(int i = 0; i < vec.size(); ++i){
				q.push(vec[i]);
			}

			int cnt = 0;
			while(cnt != q.size()){
				int i = q.front();
				q.pop();
				if(b.place(i) != board::legal){
					q.push(i);
					cnt++;
				}else{
					cnt = 0;
				}
			}

			if(b.info().who_take_turns == board::white){
				return board::black;
			}else{
				return board::white;
			}
		}

		std::vector<int> all_space(std::default_random_engine& engine){
			std::vector<int> vec;
			for(int i = 0; i < 81; ++i){
				vec.push_back(i);
			}
			std::shuffle(vec.begin(), vec.end(), engine);
			return vec;
		}

		void back_propagate(std::vector<node*>& path, unsigned winner){
			for(int i = 0; i < path.size(); ++i){
				path[i]->total_cnt++;
				if(winner != (path[i]->info()).who_take_turns){
					path[i]->win_cnt++;
				} else {
					path[i]->win_cnt--;
				}
			}
		}
	};

	void delete_tree(node* root){
		if(root->child.size() == 0){
			delete root;
			return ;
		}

		for(auto &c : root->child){
			if(c.second != nullptr){
				delete_tree(c.second);
			}
		}

		delete root;
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};

class heuristic {
public:

	inline static int value(board& b, board::piece_type who) noexcept {
		constexpr int EYE_SCORE = 16;
		constexpr int LIBERTY_SCORE = 1;
		constexpr int ATARI_SCORE = 8;
		constexpr int SELF_CORNER_ADJACENT_SCORE = 4;
		int score = 0;
		for (int y = 0; y < board::size_y; y++)
			for (int x = 0; x < board::size_x; x++)
				score +=
					+ EYE_SCORE                  * is_eye(b, x, y, who)
					+ LIBERTY_SCORE              * count_liberty(b, x, y)
					+ ATARI_SCORE                * is_atari(b, x, y, who)
					+ SELF_CORNER_ADJACENT_SCORE * count_corner_adjacent(b, x, y, who);
		return score;
	}

	inline static int action_value(board& b, int x, int y, board::piece_type who) noexcept {
		constexpr int EYE_SCORE = 16;
		constexpr int LIBERTY_SCORE = 8;
		constexpr int ATARI_SCORE = 24;
		constexpr int SELF_CORNER_ADJACENT_SCORE = 8;
		constexpr int OPPO_CORNER_ADJACENT_SCORE = 1;
		constexpr int OPPO_SIDE_ADJACENT_SCORE = 1;
		constexpr int BLOCK_OPPO_SCORE = 24;
		//constexpr int SELF_CAN_PLAY_SCORE = 32;
		int score =
			- EYE_SCORE                  * is_eye(b, x, y, who)
			+ LIBERTY_SCORE              * count_liberty(b, x, y)
			+ ATARI_SCORE                * is_atari(b, x, y, who)
			+ SELF_CORNER_ADJACENT_SCORE * count_corner_adjacent(b, x, y, who)
			+ OPPO_CORNER_ADJACENT_SCORE * count_corner_adjacent(b, x, y, board::piece_type(3u - who))
			+ OPPO_SIDE_ADJACENT_SCORE   * count_side_adjacent(b, x, y, board::piece_type(3u - who));

		int before_count_available_place = count_available_place(b, board::piece_type(3u - who));
		//int before_count_only_one_can_place = count_only_one_can_place(b, who);
		auto autoundo = b.temporary();
		autoundo.place(x, y, who);
		int after_count_available_place = count_available_place(b, board::piece_type(3u - who));
		//int after_count_only_one_can_place = count_only_one_can_place(b, who);

		score += EYE_SCORE * (
			+ is_eye(b, x, y - 1, who)
			+ is_eye(b, x - 1, y, who)
			+ is_eye(b, x + 1, y, who)
			+ is_eye(b, x, y + 1, who)
		);
		score += BLOCK_OPPO_SCORE    * (before_count_available_place - after_count_available_place);
		//score += SELF_CAN_PLAY_SCORE * (after_count_only_one_can_place - before_count_only_one_can_place);
		return score;
	}

	inline static bool is_eye(const board& b, int x, int y, board::piece_type who) noexcept {
		if (x < 0 || x >= board::size_x || y < 0 || y >= board::size_y || b[x][y] != board::empty) return false;
		return (x + 0 < 0 || x + 0 >= board::size_x || y - 1 < 0 || y - 1 >= board::size_y || b[x + 0][y - 1] == who || b[x + 0][y - 1] == board::hollow)
			&& (x - 1 < 0 || x - 1 >= board::size_x || y + 0 < 0 || y + 0 >= board::size_y || b[x - 1][y + 0] == who || b[x - 1][y + 0] == board::hollow)
			&& (x + 1 < 0 || x + 1 >= board::size_x || y + 0 < 0 || y + 0 >= board::size_y || b[x + 1][y + 0] == who || b[x + 1][y + 0] == board::hollow)
			&& (x + 0 < 0 || x + 0 >= board::size_x || y + 1 < 0 || y + 1 >= board::size_y || b[x + 0][y + 1] == who || b[x + 0][y + 1] == board::hollow);
	}

	inline static int count_liberty(const board& b, int x, int y) noexcept {
		return (x + 0 >= 0 && x + 0 < board::size_x && y - 1 >= 0 && y - 1 < board::size_y && b[x + 0][y - 1] == board::empty)
			 + (x - 1 >= 0 && x - 1 < board::size_x && y + 0 >= 0 && y + 0 < board::size_y && b[x - 1][y + 0] == board::empty)
			 + (x + 1 >= 0 && x + 1 < board::size_x && y + 0 >= 0 && y + 0 < board::size_y && b[x + 1][y + 0] == board::empty)
			 + (x + 0 >= 0 && x + 0 < board::size_x && y + 1 >= 0 && y + 1 < board::size_y && b[x + 0][y + 1] == board::empty);
	}

	inline static bool is_atari(const board& b, int x, int y, board::piece_type who) noexcept {
		return (x + 0 < 0 || x + 0 >= board::size_x || y - 1 < 0 || y - 1 >= board::size_y || b[x + 0][y - 1] == 3u - who || b[x + 0][y - 1] == board::hollow)
			 + (x - 1 < 0 || x - 1 >= board::size_x || y + 0 < 0 || y + 0 >= board::size_y || b[x - 1][y + 0] == 3u - who || b[x - 1][y + 0] == board::hollow)
			 + (x + 1 < 0 || x + 1 >= board::size_x || y + 0 < 0 || y + 0 >= board::size_y || b[x + 1][y + 0] == 3u - who || b[x + 1][y + 0] == board::hollow)
			 + (x + 0 < 0 || x + 0 >= board::size_x || y + 1 < 0 || y + 1 >= board::size_y || b[x + 0][y + 1] == 3u - who || b[x + 0][y + 1] == board::hollow)
			== 3 && count_liberty(b, x, y) == 1;
	}

	inline static int count_corner_adjacent(const board& b, int x, int y, board::piece_type who) noexcept {
		return (x - 1 >= 0 && x - 1 < board::size_x && y - 1 >= 0 && y - 1 < board::size_y && b[x - 1][y - 1] == who)
			 + (x + 1 >= 0 && x + 1 < board::size_x && y - 1 >= 0 && y - 1 < board::size_y && b[x + 1][y - 1] == who)
			 + (x - 1 >= 0 && x - 1 < board::size_x && y + 1 >= 0 && y + 1 < board::size_y && b[x - 1][y + 1] == who)
			 + (x + 1 >= 0 && x + 1 < board::size_x && y + 1 >= 0 && y + 1 < board::size_y && b[x + 1][y + 1] == who);
	}

	inline static int count_side_adjacent(const board& b, int x, int y, board::piece_type who) noexcept {
		return (x + 0 >= 0 && x + 0 < board::size_x && y - 1 >= 0 && y - 1 < board::size_y && b[x + 0][y - 1] == who)
			 + (x - 1 >= 0 && x - 1 < board::size_x && y + 0 >= 0 && y + 0 < board::size_y && b[x - 1][y + 0] == who)
			 + (x + 1 >= 0 && x + 1 < board::size_x && y + 0 >= 0 && y + 0 < board::size_y && b[x + 1][y + 0] == who)
			 + (x + 0 >= 0 && x + 0 < board::size_x && y + 1 >= 0 && y + 1 < board::size_y && b[x + 0][y + 1] == who);
	}

	inline static int count_only_one_can_place(const board& b, board::piece_type who) noexcept {
		int count = 0;
		for (int y = 0; y < board::size_y; y++)
			for (int x = 0; x < board::size_x; x++)
				count += (b.test(x, y, who) == board::legal) && (b.test(x, y, 3u - who) != board::legal);
		return count;
	}

	inline static int count_available_place(const board& b, board::piece_type who) noexcept {
		int count = 0;
		for (int y = 0; y < board::size_y; y++)
			for (int x = 0; x < board::size_x; x++)
				count += b.test(x, y, who) == board::legal;
		return count;
	}

};

class heuristic_agent : public random_agent {
public:
	heuristic_agent(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state, int steps) {
		
		if (who == board::black && state.test(4, 4, who) == board::legal)
			return action::place(40, who);
		
		std::shuffle(space.begin(), space.end(), engine);
		int max_index = -1;
		int max_score = -9999;
		board after = state;
		for (int i = 0, end = space.size(); i < end; i++) {
			const action::place& move = space[i];
			board::point p = move.position();
			if (state.test(p.x, p.y, who) == board::legal) {
				int score = heuristic::action_value(after, p.x, p.y, who);
				if (score > max_score) {
					max_index = i;
					max_score = score;
				}
			}
		}
		
		auto i = space[max_index].position().i;
		if (!(i == 4 || i == 36 || i == 44 || i == 76)) {
			if (state.test(4, 4, who) == board::legal)
				return action::place(40, who);
		}
		
		return max_index == -1 ? action() : space[max_index];
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};
