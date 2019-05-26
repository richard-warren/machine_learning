prefix = "F:\\cells_kitchen_files\\images_"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

vid_num = 5

preview_vid(prefix+suffixes[vid_num], frames_to_show=50, fps=100, close_when_done=False, percentiles=(0, 100))
