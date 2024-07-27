

from io import StringIO
import os 
import sys 

file_name = sys.argv[2]
detect_version = sys.argv[1]

if detect_version == 'yolo':
    import yoloUtils as Utils
elif detect_version == 'mrcnn':
    import mrcnnUtils as Utils
else:
    print('wrong type')
    exit()

pre_dir = "./"
stt = 'test'

#file_name = 'Understanding the Representation Power of Graph Neural Nets in Learning Graph Topology.mp4'

print(file_name)
model = Utils.load_model()

# Appending app path to upload folder path within app root folder

Utils.main_solution(model,pre_dir,file_name)
file_name_md = file_name.split('.')[0]
print("path  :" ,file_name)
file_loc = os.path.join(os.getcwd() ,f'{file_name_md}.md')
print("UPLOADS : ",file_loc)
#file_loc
#, mimetype = 'text/md',attachment_filename='PLEASE.md',
#as_attachment = True)
# Returning file from appended path
#return send_from_directory(directory=current_app.root_path , filename= path)

