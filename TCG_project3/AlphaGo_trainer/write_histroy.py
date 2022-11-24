import os

def write_one_file(route):
	#fp = open(target_dir + '/P1B_P2W.log', 'r')
	fp = open(route, 'r')
	lines = fp.readlines()
	fp.close()

	m = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8}

	f_h = open('history.txt', 'a')
	f_w = open('win_or_loss.txt', 'a')
	for line in lines:
		li = line.strip('\n').split()
		if len(li) > 2 and li[2] == "resign":
			f_h.write('\n')
			if li[0][0] == "B":
				f_w.write("-1\n")  # loss
			else:
				f_w.write("1\n")  # win

		if len(li) > 1 and li[1] == "play":
			pos = int(m[li[3][0]]) * 9 + int(li[3][1]) - 1
			if li[2] == "B":
				f_h.write(str(pos) + ' ')
			else:
				f_h.write(str(pos + 81) + ' ')


	f_h.close()
	f_w.close()


arr = os.listdir()
target_dir = []
for a in arr:
	tmp = a.split('-')
	if len(tmp) == 3 and tmp[0] == "gogui" and tmp[1] == "twogtp":
		target_dir.append(a)

for tar_dir in target_dir:
	write_one_file(tar_dir + '/P1B_P2W.log')
	write_one_file(tar_dir + '/P2B_P1W.log')

