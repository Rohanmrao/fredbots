#!/usr/bin/python3

import rospy
from fredbots.srv import AddTwoInts
from fredbots.srv import AddTwoIntsResponse



def handle_add_two_ints(req):
    prev_x = req.prev_x
    prev_y = req.prev_y

    cur_x = req.cur_x
    cur_y = req.cur_y

    next_x = req.next_x
    next_y = req.next_y

    print("current from client: ", cur_x, cur_y)
    print("next from client: ", next_x, next_y)
    grid[cur_x][cur_y] = 1
    
    for i in range(21):
        for j in range(21):
            if prev_x != -1 and prev_y != -1:
                    grid[prev_x][prev_y] = 0
            if grid[next_x][next_y] == 1:
                # [TO PRINT]
                for i in range(21):
                    for j in range(21):
                        print(grid[i][j], end='\t')
                    print('\n')
                print("\n\n")
                return AddTwoIntsResponse(1)
            else:
                # grid[cur_x][cur_y] = 0
                grid[next_x][next_y] = 1
                # [TO PRINT]
                for i in range(21):
                    for j in range(21):
                        print(grid[i][j], end='\t')
                    print('\n')
                print("\n\n")
                return AddTwoIntsResponse(0)
    

def contr_server():

    rospy.init_node('add_two_ints_server')
    service = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    rospy.spin()


def main():
    global grid 
    grid = [[0 for j in range(21)] for i in range(21)]

    # for i in range(21):
    #     for j in range(21):
    #         grid[i][j] = 0
    #         if i==0 and j ==1:
    #             grid[i][j] = 1
    #         if i==1 and j ==0:
    #             grid[i][j] = 1
    #         if i==1 and j ==2:
    #             grid[i][j] = 1

    # [TO PRINT]
    # for i in range(21):
    #     for j in range(21):
    #         print(grid[i][j], end='\t')
    #     print('\n')
    # print("\n\n")
    
    
    contr_server()


if __name__ == '__main__':
     main()


