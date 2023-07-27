obs = [(2,3), (2,4), (2,5), (5,0), (5,1), (5,2)]
grid = [[0 for j in range(6)] for i in range(6)]


for i in range(6):
    for j in range(6):
        grid[i][j] = 0
        
        # [DEFINING OBSTACLES]
        if (i,j) in obs:
            grid[i][j] = 1

for i in range(6):
    for j in range(6):
        print(grid[i][j], end='\t')
    print('\n')
