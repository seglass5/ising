# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:32:14 2019

@author: thecl
"""
"""
def largest_connected_component(nrows, ncols, grid):
    ""#Find largest connected component of 1s on a grid.
    """
"""
    def traverse_component(pos):
    """    """Returns no. of unseen elements connected to (i,j).""""""
        i, j = pos
        result = 1

        # Check all four neighbours
        for new_pos in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if new_pos in elements:
                elements.remove(new_pos)
                result += traverse_component(new_pos)
        return result

    # Tracks size of largest connected component found
    elements = set((i, j) for i, line in enumerate(grid) for j, cell in enumerate(line) if cell)
    largest = 0
    while elements:
        pos = elements.pop()
        largest = max(largest, traverse_component(pos))
    return largest
"""
import numpy as np
#from random import randint

N = 5
spin = np.array([1,-1])
grid = np.random.choice(spin, size=(N,N))
print(grid)


def largest_connected_component(nrows, ncols, grid):
    """Find largest connected component of 1s on a grid."""

    def traverse_component(i, j):
        """Returns no. of unseen elements connected to (i,j)."""
        grid[i][j] = -1
        result = 1
        #print('iandj')
        #print(i, j)

        # Check all four neighbours
        if i > 0 and (grid[i-1][j]==1):
            result += traverse_component(i-1, j)
            #print(i,j)
        if j > 0 and (grid[i][j-1]==1):
            result += traverse_component(i, j-1)
            #print(i,j)
        if i < len(grid)-1 and (grid[i+1][j]==1):
            result += traverse_component(i+1, j)
            #print(i, j)
        if j < len(grid[0])-1 and (grid[i][j+1]==1):
            result += traverse_component(i, j+1)
            #print(i,j)
        return result

    # Tracks size of largest connected component found
    component_size = 0

    for i in range(nrows):
        for j in range(ncols):
            if (grid[i][j] == 1):
                component_size = max(component_size, traverse_component(i,j))

    return component_size
print(largest_connected_component(5,5,grid))