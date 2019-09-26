import pyorthogonal_polynomials as pyortho
fortho =  pyortho.orthogonal_polynomials
z = 1.01
for d in range(45):
    print(d,fortho.hermite(z,d))

