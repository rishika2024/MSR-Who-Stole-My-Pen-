from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
# The robot object is what you use to control the robot

""" home ee position = [[ 0.96043053  0.          0.27851966  0.14650298]
 [ 0.          1.          0.          0.        ]
 [-0.27851966  0.          0.96043053  0.16157735]
 [ 0.          0.          0.          1.        ]]"""


#sleep ee position = [0.0, 0.02454369328916073, 0.06902913749217987, 0.03528155758976936]


robot = InterbotixManipulatorXS("px100", "arm", "gripper")


robot_startup()
mode = 'h'
# Let the user select the position
while mode != 'q':
    mode=input("[h]ome, [s]leep, [q]uit, [r]elease, [g]rasp, [t]rajectory, [p]ose [a]pos ")
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
        print(robot.core.robot_get_joint_states())

    elif mode == 'a':
        robot.arm.set_single_joint_position('waist', 1.57)


    

robot_shutdown()
