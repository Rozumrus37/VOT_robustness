import subprocess
import os
import compute_iou
from compute_iou import *

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the command '{command}': {e}")
        exit(1)


def execute_commands(commands):
    for command in commands:
        print(command, " executed!")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{command}': {e}")
            exit(1)

def replace_first_line(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1:
            first_line = file1.readline().strip()

        with open(file2_path, 'r') as file2:
            lines = file2.readlines()
        
        lines[0] = first_line + '\n'

        with open(file2_path, 'w') as file2:
            file2.writelines(lines)

        print("Replacement completed successfully.")
    except Exception as e:
        print(f"Error: {e}")

def replace_lines_except_first(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1:
            lines_to_replace = file1.readlines()[1:]

        with open(file2_path, 'r') as file2:
            lines = file2.readlines()

        lines[1:] = lines_to_replace

        with open(file2_path, 'w') as file2:
            file2.writelines(lines)

        print("Replacement completed successfully.")
    except Exception as e:
        print(f"Error: {e}")


def write_to_file(file_name, seq_name):
    with open(file_name, 'w') as file:
        file.write(seq_name + '\n')

if __name__ == "__main__":
    seqs = ['book', 'agility', 'animal', 'ants1', 'ball3', 'basketball', 'birds1', 'birds2', 'snake', 'bag', 'polo']

    for seq in seqs:
        cmds = ['rm ' + '/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/sequences/list.txt']
        write_to_file('/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/sequences/list.txt', seq)

        base_path = os.path.join('/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/sequences', seq)
        params = [[0, 0, 0], [10, 1, 18], [9, 1, 20], [8, 1, 23], [80, 12, 23], [2, 1, 70]]

        for param in params:
            if param == [0, 0, 0]:
                cmds = []
                if os.path.exists(os.path.join(base_path, 'color')):
                    rm_color = 'rm -r ' + os.path.join(base_path, 'color')
                    cmds.append(rm_color)

                color_name = 'init_color'
                cp_color = 'cp -r ' + os.path.join(base_path, color_name) + ' ' + os.path.join(base_path, 'color')

                cmds.append(cp_color)
                execute_commands(cmds)
                cmds = []

                if os.path.exists(os.path.join(base_path, 'groundtruth.txt')):
                    rm_gt = 'rm ' + os.path.join(base_path, 'groundtruth.txt')
                    cmds.append(rm_gt)

                cp_groundt = 'cp ' + os.path.join(base_path, 'gt_init.txt') + ' ' + os.path.join(base_path, 'groundtruth.txt')
                cmds.append(cp_groundt)

                execute_commands(cmds)

                python_program_path = 'CUDA_VISIBLE_DEVICES=3, python3 -m vot evaluate --workspace $(pwd) DAMTMask'
                run_command(python_program_path)

                baseline_res_path = '/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/results/DAMTMask/baseline'
                setup_path = os.path.join(baseline_res_path, 'setup_' + seq + '_99')
                
                if not os.path.exists(setup_path):
                    os.mkdir(setup_path)

                mv_book = 'mv ' + os.path.join(baseline_res_path, seq) + ' ' + os.path.join(baseline_res_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]))
                mv_to_setup = 'mv ' + os.path.join(baseline_res_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2])) + ' ' + setup_path
                cp_gt = 'cp ' + os.path.join(base_path, 'groundtruth.txt') + ' ' + os.path.join(setup_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_gt.txt')
                cp_color_to_setup = 'cp -r ' + os.path.join(base_path, 'color') + ' ' + os.path.join(setup_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_color')

                cmds.append(mv_book)
                cmds.append(mv_to_setup)
                cmds.append(cp_gt)
                cmds.append(cp_color_to_setup)
                execute_commands(cmds)
                cmds = []
                iou = get_iou("99",  str(param[0]), str(param[1]), str(param[2]), seq)
                write_to_file(os.path.join(baseline_res_path, "99" + '_' + str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_' + seq + '.txt'), str(iou))
                continue



            for fr_setup in range(0, 3):
                cmds = []

                if os.path.exists(os.path.join(base_path, 'color')):
                    rm_color = 'rm -r ' + os.path.join(base_path, 'color')
                    cmds.append(rm_color)

                if fr_setup == 0:
                    color_name = str(param[0]) + 'l_' + str(param[1]) + 'th_' + str(param[2]) + 'p_color'
                    cp_color = 'cp -r ' + os.path.join(base_path, color_name) + ' ' + os.path.join(base_path, 'color')

                    cmds.append(cp_color)
                    execute_commands(cmds)
                    cmds = []

                    if os.path.exists(os.path.join(base_path, 'color', '00000001.jpg')):
                        rm_1_frame = 'rm ' + os.path.join(base_path, 'color', '00000001.jpg')
                        cmds.append(rm_1_frame)

                    cp_1_frame = 'cp ' + os.path.join(base_path, 'init_color', '00000001.jpg') + ' ' + os.path.join(base_path, 'color')
                    cmds.append(cp_1_frame)

                    if os.path.exists(os.path.join(base_path, 'groundtruth.txt')):
                        rm_gt = 'rm ' + os.path.join(base_path, 'groundtruth.txt')
                        cmds.append(rm_gt)
                    gen_new_segm = 'python3 /home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/gen_new_segm.py ' + seq + ' ' + str(param[0]) + ' ' + str(param[1])
                    cmds.append(gen_new_segm)

                    execute_commands(cmds)
                    cmds = []
                    replace_first_line(os.path.join(base_path, 'gt_init.txt') , os.path.join(base_path, 'groundtruth.txt'))
                elif fr_setup == 1:
                    color_name = str(param[0]) + 'l_' + str(param[1]) + 'th_' + str(param[2]) + 'p_color'
                    cp_color = 'cp -r ' + os.path.join(base_path, "init_color") + ' ' + os.path.join(base_path, 'color')

                    cmds.append(cp_color)
                    execute_commands(cmds)
                    cmds = []

                    rm_1_frame = 'rm ' + os.path.join(base_path, 'color', '00000001.jpg')
                    cp_1_frame = 'cp ' + os.path.join(base_path, color_name, '00000001.jpg') + ' ' + os.path.join(base_path, 'color')
                    rm_gt = 'rm ' + os.path.join(base_path, 'groundtruth.txt')
                    gen_new_segm = 'python3 /home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/gen_new_segm.py ' + seq + ' ' + str(param[0]) + ' ' + str(param[1])
                    
                    cmds.append(rm_1_frame)
                    cmds.append(cp_1_frame)
                    cmds.append(rm_gt)
                    cmds.append(gen_new_segm)
                    execute_commands(cmds)
                    cmds = []
                    replace_lines_except_first(os.path.join(base_path, 'gt_init.txt') , os.path.join(base_path, 'groundtruth.txt')) 
                else:
                    color_name = str(param[0]) + 'l_' + str(param[1]) + 'th_' + str(param[2]) + 'p_color'
                    cp_color = 'cp -r ' + os.path.join(base_path, color_name) + ' ' + os.path.join(base_path, 'color')

                    cmds.append(cp_color)
                    execute_commands(cmds)
                    cmds = []

                    rm_gt = 'rm ' + os.path.join(base_path, 'groundtruth.txt')
                    gen_new_segm = 'python3 /home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/gen_new_segm.py ' + seq + ' ' + str(param[0]) + ' ' + str(param[1])
                    
                    cmds.append(rm_1_frame)
                    cmds.append(cp_1_frame)
                    cmds.append(rm_gt)
                    cmds.append(gen_new_segm)
                    execute_commands(cmds)
                    cmds = []

                python_program_path = 'CUDA_VISIBLE_DEVICES=3, python3 -m vot evaluate --workspace $(pwd) DAMTMask'
                run_command(python_program_path)

                baseline_res_path = '/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/results/DAMTMask/baseline'
                setup_path = os.path.join(baseline_res_path, 'setup_' + seq + '_' + str(fr_setup))
                
                if not os.path.exists(setup_path):
                    os.mkdir(setup_path)

                mv_book = 'mv ' + os.path.join(baseline_res_path, seq) + ' ' + os.path.join(baseline_res_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]))
                mv_to_setup = 'mv ' + os.path.join(baseline_res_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2])) + ' ' + setup_path
                cp_gt = 'cp ' + os.path.join(base_path, 'groundtruth.txt') + ' ' + os.path.join(setup_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_gt.txt')
                cp_color_to_setup = 'cp -r ' + os.path.join(base_path, 'color') + ' ' + os.path.join(setup_path, str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_color')

                cmds.append(mv_book)
                cmds.append(mv_to_setup)
                cmds.append(cp_gt)
                cmds.append(cp_color_to_setup)
                execute_commands(cmds)
                cmds = []

                iou = get_iou(str(fr_setup),  str(param[0]), str(param[1]), str(param[2]), seq)
                write_to_file(os.path.join(baseline_res_path, str(fr_setup) + '_' + str(param[0]) + '_' + str(param[1]) + '_' + str(param[2]) + '_' + seq + '.txt'), str(iou))



print("Script completed successfully.")
