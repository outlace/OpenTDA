def reduce_matrix(matrix):
    #Returns [reduced_matrix, rank, nullity]
    if np.size(matrix)==0:
        return [matrix,0,0]
    m=matrix.shape[0]
    n=matrix.shape[1]

    def _reduce(x):
        #We recurse through the diagonal entries.
        #We move a 1 to the diagonal entry, then
        #knock out any other 1s in the same  col/row.
        #The rank is the number of nonzero pivots,
        #so when we run out of nonzero diagonal entries, we will
        #know the rank.
        nonzero=False
        #Searching for a nonzero entry then moving it to the diagonal.
        for i in range(x,m):
            for j in range(x,n):
                if matrix[i,j]==1:
                    matrix[[x,i],:]=matrix[[i,x],:]
                    matrix[:,[x,j]]=matrix[:,[j,x]]
                    nonzero=True
                    break
            if nonzero:
                break
        #Knocking out other nonzero entries.
        if nonzero:
            for i in range(x+1,m):
                if matrix[i,x]==1:
                    matrix[i,:] = np.logical_xor(matrix[x,:], matrix[i,:])
            for i in range(x+1,n):
                if matrix[x,i]==1:
                    matrix[:,i] = np.logical_xor(matrix[:,x], matrix[:,i])
            #Proceeding to next diagonal entry.
            return _reduce(x+1)
        else:
            #Run out of nonzero entries so done.
            return x
    rank=_reduce(0)
    return [matrix, rank, n-rank]

# Source: < https://triangleinequality.wordpress.com/2014/01/23/computing-homology/ >
