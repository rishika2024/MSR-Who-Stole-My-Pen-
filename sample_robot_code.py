from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
import math
# The robot object is what you use to control the robot

""" home ee position = 
[[ 0.99090264  0.          0.13458071  0.24967073]
 [ 0.          1.          0.          0.        ]
 [-0.13458071  0.          0.99090264  0.16721934]
 [ 0.          0.          0.          1.        ]]"""


"""sleep ee position =
 [[ 0.90788615  0.          0.41921681  0.09038335]
 [ 0.          1.          0.          0.        ]
 [-0.41921681  0.          0.90788615  0.08427828]
 [ 0.          0.          0.          1.        ]]"""



robot = InterbotixManipulatorXS("px100", "arm", "gripper")


robot_startup()
mode = 'h'
# Let the user select the position
"""while mode != 'q':
    mode=input("[h]ome, [s]leep, [q]uit, [r]elease, [g]rasp, [t]rajectory, [p]ose, [a]rotate_right, [l]rotate_left")
    if mode == "h":
        robot.arm.go_to_home_pose()
        print(robot.arm.get_ee_pose())
        
    elif mode == "s":
        robot.arm.go_to_sleep_pose()
        print(robot.arm.get_ee_pose())

    elif mode == "r":
        robot.gripper.release()
        #print(robot.arm.get_ee_pose())

    elif mode == 'g':
        robot.gripper.grasp()
        #print(robot.arm.get_ee_pose())
    
    elif mode == 't':
        robot.arm.set_ee_pose_components(x=0.09, y=0, z=0.17, moving_time =2)
        print(robot.arm.get_ee_pose())

    elif mode =='p':
        print(robot.arm.get_ee_pose())

    elif mode == 'a':
        robot.arm.set_single_joint_position('waist', 1.57)
        print(robot.arm.get_ee_pose())

    elif mode == 'l':
        robot.arm.set_single_joint_position('waist', -1.57)
        print(robot.arm.get_ee_pose())"""

x=0.2496
y=0
z=0.1672
angle = 1.57


while mode != 'q':

    mode=input("[n]ext, [q]uit, [g]rasp, [r]elease")

    if mode == 'n':

        robot.arm.set_ee_pose_components(x=0.2496, y=0, z=0.1672, moving_time =2)
        robot.arm.set_single_joint_position('waist', angle)
        print(f"robot pose ={robot.arm.get_ee_pose()}")
        x-=0.01
        z+=0.01
        angle-=0.1

    elif mode == 'r':
        robot.gripper.release()

    elif mode == 'g':
        robot.gripper.grasp()








    

robot_shutdown()
