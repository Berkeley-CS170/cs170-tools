
def levenshtein(a, b):
    """
    Compute the Levenshtein distance between a and b.
    Good thing I took 170, or else I wouldn't know how to implement this.
    """

    M = {}

    def recur(i, j):
        if i == 0: ans = j
        elif j == 0: ans = i
        else:
            if a[i] == b[j]: ans = M[(i-1, j-1)]
            else:
                ans = min(
                    M[(i-1, j-1)] + 1, # substitute
                    M[(i, j-1)] + 1, # addition
                    M[(i-1, j)] + 1 # deletions
                )
        M[(i, j)] = ans

    for i in range(len(a)):
        for j in range(len(b)):
            recur(i, j)
    
    return M[len(a) - 1, len(b) - 1]
