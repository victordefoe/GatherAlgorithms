
import subprocess
import os
from os.path import join as opj
import datetime, time
import shutil

inp = r'H:\liumian\not_sink_egg_cut_press.mp4'

inp_dir = r'H:\liumian'
# subprocess.call('ffmpeg -i {} -strict -2  -qscale 0 -intra {}'.format(inp, opj(inp_dir, 'ns.mp4')))
# inp = opj(inp_dir, 'ns.mp4')
outp = r'H:\liumian\not_sink_egg_cut_press_.mp4'
tmp_cuts = r'H:\\liumian\\not_sink\\'
shutil.rmtree(tmp_cuts)
if not os.path.exists(tmp_cuts):
    os.mkdir(tmp_cuts)





# cut_pair = [('00:00', '00:22'), 
#             ('00:29','00:35'),
#             ('00:36','00:38'), 
#             ('00:43','00:49'),
#             ('00:50','00:56'),
#             ('01:01','01:05'),
#             ('01:10','01:20'),
#             ('01:34','01:55')]


# def str2sec(x):
#     '''
#     字符串时分秒转换成秒
#     '''
#     m, s = x.strip().split(':') #.split()函数将其通过':'分隔开，.strip()函数用来除去空格
#     return int(m)*60 + int(s) #int()函数转换成整数运算

# cut_pair = [(pair[0],str2sec(pair[1]) - str2sec(pair[0])) for pair in cut_pair ]

# for index, each in enumerate(cut_pair):
#     print(each)
#     command = "ffmpeg -ss {} -t {} ".format(each[0], each[1]) + \
#                 ' -i ' + inp + ' -vcodec copy -acodec copy '+ opj(tmp_cuts, '{}.mp4'.format(index))
#     print(command)
#     subprocess.call(command)

# # l = [opj(tmp_cuts, x) for x in os.listdir(tmp_cuts)]
# list_file = opj(inp_dir, 'list.txt')
# with open(list_file, 'w') as f:
#     for x in os.listdir(tmp_cuts):
#         f.writelines('file '+ opj(tmp_cuts, x) + '\n')

# merge_cmd = 'ffmpeg -f concat -safe 0 -i {} -c copy {}'.format(list_file, outp)
# subprocess.call(merge_cmd)

# subprocess.call('ffmpeg -i {} -b 600k {}'.format(inp, outp))
subprocess.call('ffmpeg -i {} -vf "transpose=2" {}'.format(inp, outp))

