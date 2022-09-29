/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
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
#include "weight.h"
#include <unistd.h>

struct state {
	board board_before;
	board board_after;
	int reward;
	float value;
	state(){
		reward = 0;
		value = 0;
	}
};

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
	virtual action take_action(const board& b, float& state_value, int& r) { return action(); }
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
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		/*
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
		*/
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);

	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after, float& state_value, int& r) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class my_slider : public weight_agent {
public:
	my_slider(const std::string& args = "") : weight_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }), space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }) {}

	virtual action take_action(const board& before, float& state_value, int& r) {
		float best_v = -std::numeric_limits<float>::max();
		int best_reward = -std::numeric_limits<float>::max();
		int best_state_value = -std::numeric_limits<float>::max();
		int best_op = -1;
		for (int op : opcode) {
			board tmp = board(before);
			board::reward reward = action::slide(op).apply(tmp);
			if (reward == -1) {
				continue;
			}

			float vs = estimate_value(tmp);
			float v = reward + vs;
			if (v > best_v) {
				best_v = v;
				best_op = op;
				best_reward = reward;
				best_state_value = vs;
			}

		}
		
		if (best_op == -1) {
			return action();
		} else {
			state_value = best_state_value;
			r = best_reward;
			return action::slide(best_op);
		}
		
	}

	float estimate_value(const board& b) {
		float sum = 0.0;
		board tmp = board(b);
		
		for (int i = 0; i < 4; ++i) {
			int idx0 = extract_feature(tmp, 0, 1, 2, 3);
			int idx1 = extract_feature(tmp, 4, 5, 6, 7);
			int idx2 = extract_feature(tmp, 8, 9, 10, 11);
			int idx3 = extract_feature(tmp, 12, 13, 14, 15);
			sum += (net[0][idx0] + net[1][idx1] + net[2][idx2] + net[3][idx3]);
		}
		
		return sum;

	}

	float adjust_value(const board& b, float target) {
		return 0;
	}

	int extract_feature(const board& after, int a, int b, int c, int d) const {
		return after(a) * 16 * 16 * 16 + after(b) * 16 * 16 + after(c) * 16 + after(d);
	}

	void update(std::vector<state>& path) {
		float tmp = 0;
		for (int i = path.size() - 1; i >= 0; i--) {
			float td_error = tmp - path[i].value;
			tmp = path[i].reward + adjust_value(path[i].board_after, alpha * td_error);
		}
	}

private:
	std::array<int, 4> opcode;
	std::vector<int> space;
};
