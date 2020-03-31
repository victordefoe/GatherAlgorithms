for index, each in enumerate(cut_pair):
#     print(each)
#     command = "ffmpeg -ss {} -t {} ".format(each[0], each[1]) + \
#                 ' -i ' + inp + ' -vcodec copy -acodec copy '+ opj(tmp_cuts, '{}.mp4'.format(index))
#     print(command)
#     subprocess