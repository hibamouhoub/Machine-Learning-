
# method 1

i = int(input('Entrez le nombre de soldats : '))
l = list(range(1,i+1,1))
l2 = []
nombre_de_sauts = int(input('Entrez le nombre de sauts: '))
i = nombre_de_sauts
while l != []:
    if len(l) == 1:
        l2.append(l.pop(0))
    else: 
        if len(l)-i<nombre_de_sauts:
            b =nombre_de_sauts-(len(l) - i)
        else:
            b = i+(nombre_de_sauts-1)
        if b>=len(l)-1:
            b= b%(len(l)-1)
        l2.append(l.pop(i))
        i = b
    
    
# method 2
