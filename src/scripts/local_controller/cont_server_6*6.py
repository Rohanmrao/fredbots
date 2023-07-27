#!/usr/bin/env python

import rospy
from fredbots.srv import AddTwoInts
from fredbots.srv import AddTwoIntsResponse



def handle_add_two_ints(req):
    cur_x = req.cur_x
    cur_y = req.cur_y

    next_x = req.next_x
    next_y = req.next_y

    grid[cur_x][cur_y] = 1
    
    print("current from server: ", cur_x, cur_y)
    print("next from server: ", next_x, next_y)

    for i in range(6):
        for j in range(6):
            if grid[next_x][next_y] == 1:
                # [TO PRINT]
                for i in range(6):
                    for j in range(6):
                        print(grid[i][j], end='\t')
                    print('\n')
                print("\n\n")
                return AddTwoIntsResponse(1)
            else:
                grid[cur_x][cur_y] = 0
                grid[next_x][next_y] = 1
                # [TO PRINT]
                for i in range(6):
                    for j in range(6):
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
    grid = [[0 for j in range(6)] for i in range(6)]

    obs = [(2,3), (2,4), (2,5), (5,0), (5,1), (5,2)]

    for i in range(6):
        for j in range(6):
            grid[i][j] = 0
            
            # [DEFINING OBSTACLES]
            if (i,j) in obs:
                grid[i][j] = 1

    # [TO PRINT]
    for i in range(6):
        for j in range(6):
            print(grid[i][j], end='\t')
        print('\n')
    print("\n\n")
    
    
    contr_server()


if __name__ == '__main__':
     main()


