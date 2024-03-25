from numba import njit

@njit
def rec(i, stack, stack_len, birth, death, out, max_length, G):
    if birth <= death:
        return
    if stack_len > 2 and G[i, stack[0]] > death:
        out.append((stack_len, min(birth, G[i, stack[0]]), death))
    if stack_len >= max_length:
        return
    if stack_len > 2:
        death = max(death, G[i, stack[0]])
    for j in range(stack[0] + 1, G.shape[0]):
        ok = True
        for k in range(stack_len):
            ok &= stack[k] != j
        if not ok:
            continue
        death1 = death
        for k in range(1, stack_len - 1):
            death1 = max(death1, G[j, stack[k]])
        stack[stack_len] = j
        rec(j, stack, stack_len + 1, min(birth, G[i, j]), death1, out, max_length, G)

@njit(parallel=True)
def fast_hole_analysis(G, max_length):
    ans = [(-1, 0.0, 0.0)]
    stack = [0] * max_length
    for i in range(G.shape[0]):
        stack[0] = i
        rec(i, stack, 1, 1.0, 0.0, ans, max_length, G)
    return ans[1:]