

N = 3

for r in range(N):
    for c in range(r+1):
        B = 'B{}'.format(c)
        C = 'C{}'.format(r+1)
        A = ''
        for i in range(c+1, r+1):
            A = 'A{}'.format(i) + A
        print(C + A + B, end=' ')
    print()
