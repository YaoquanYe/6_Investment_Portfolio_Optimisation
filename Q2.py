
# coding: utf-8

# In[1]:


import numpy as np
import copy
# import random


# In[2]:


file=open("./datasets/assets1.txt")  # C:\\Users\\Daniel\\Desktop\\
lines=file.readlines() 

# 读取 收益与标准差
assert_num = np.int(lines[0].strip().split(' ')[0])
RetAndDev = np.ones([assert_num, 2])
for i in range(1, assert_num+1):
    line = lines[i]
    line=line.strip().split(' ')
    ret = np.float64(line[0], dtype=np.float)
    stddev = np.float64(line[1], dtype=np.float)
    RetAndDev[i-1, 0] = ret
    RetAndDev[i-1, 1] = stddev
    
# 读取 correlation
correlation = np.ones([assert_num, assert_num])
for i in range(assert_num+1, assert_num+1+int((1+assert_num)*assert_num/2)):
    line = lines[i]
    line=line.strip().split(' ')
    correlation[np.int(line[0])-1, np.int(line[1])-1] = np.float64(line[2])
    correlation[np.int(line[1])-1, np.int(line[0])-1] = np.float64(line[2])


# In[3]:


def eval(solution, renda):
    Q = solution['assert']
    s = solution['prop']
    
    F = 1 - omega*num_to_select
    L = np.sum(s)    
#     print("L:", L, "F:", F, "s：", s)
    w = s*F/L + 0.01
#         print("w:", w)

    index = 0
    prop = np.zeros(assert_num)

    for i in range(assert_num):
        if (i in Q):
            prop[i] = w[index]
            index += 1
        else:
            prop[i] = 0

    R = np.sum(prop * RetAndDev[:, 0])
#     print("R:", R)
    CoVar = 0
    for i in range(assert_num):
        for j in range(assert_num):
            tmp = prop[i] * prop[j] * correlation[i, j] * RetAndDev[i, 0] * RetAndDev[i, 1]
            CoVar += tmp
#     print("CoVar:", CoVar)

    f = renda*CoVar - (1-renda)*R
#     print("f:", f)

    R_list.append(R)
    CoVar_list.append(CoVar)
    
    global max_f, max_f, max_CoVar
    if f > max_f:
        max_f = f
        max_R = R
        max_CoVar = CoVar

    global min_f, min_f, min_CoVar, H, bestS
    if f < min_f:
#         print("min")
#         print("min")
#         print("min")
        min_f = f
        min_R = R
        min_CoVar = CoVar
        H.append({"assert": Q, "prop": s, "f":f})
        bestS = {"assert": Q, "prop": s, "f":f}
    return f


# In[4]:


num_of_evaluation = assert_num*100
num_of_random = 30
num_to_select = 10
renda = 0.5
omega = 0.01
H  = []
bestS = {}

max_f = 0
max_R = 0
max_CoVar = 0
min_f = 10000000000000000000
min_R = 0
min_CoVar = 0

R_list = []
CoVar_list = []

alpha = 0.95
init_times = 1000
T = 0
T_star = 10
iterNum_of_T = 10


# In[ ]:


while num_of_evaluation:# init
    while num_of_random:
#         print("num_of_evaluation:", num_of_evaluation, "num_of_random:", num_of_random)
        
        np.random.seed(num_of_random)
        
        init_times = 1000
        while init_times:    
            # S
            items_tmp = np.zeros([assert_num])
            items_tmp[:num_to_select] = 1
            np.random.shuffle(items_tmp)
            Q = []
            for i in range(assert_num):
                if items_tmp[i] == 1:
                    Q.append(i)
        #     print("Q:", Q, type(Q))
            s = np.random.randint(100, size=num_to_select)
        #     print("s:", s, type(s))

            solu = {'assert': Q, 'prop': s}
            eval(solu, renda)

        #     print(init_times)
            init_times -= 1  
        
        R_list = R_list[:-1000]
        CoVar_list = CoVar_list[:-1000]

        global bestS
        T = abs(bestS["f"]/10)
        for i in range(T_star):  # 
            for j in range(iterNum_of_T):  # 
                print()
                print("num_of_evaluation:", num_of_evaluation, "num_of_random:", num_of_random, "SA No:", i, " ", j)
                
                assert_selected = np.random.randint(len(bestS["assert"]))
                m = 0
                if np.random.randint(100)%2:
                    m = 1
                else:
                    m = 2

                C = copy.deepcopy(bestS)

                if m == 1:
                    C["prop"][assert_selected] = 0.9*(omega + C["prop"][assert_selected]) - omega
                elif m == 2:
                    C["prop"][assert_selected] = 1.1*(omega + C["prop"][assert_selected]) - omega


                if C["prop"][assert_selected]  < 0:
                    print("有增减资产！！！", C)
                    not_R = list(set(list(range(assert_num))).difference(set(bestS["assert"]))) 
                    assert_to_add = not_R[np.random.randint(len(not_R))]
                    idx_to_del = C["assert"].index[assert_selected]
                    print(idx_to_del)

                    print(C["assert"])
                    C["assert"].remove(assert_selected)
                    print(C["assert"])
                    C["prop"] = np.array(list(C["prop"]).remove(C["prop"][idx_to_del]))

                    C["assert"].apppend(assert_to_add)
                    C["prop"] = np.array(list(C["assert"]).apppend(0))
                    print(C)

                C["f"] = eval(C ,renda)
                print("f of bestS:", bestS["f"])

                if C["f"] <= bestS['f']:
                    # 已经在eval里面更新了bestS
                    print("Better")
                    print("Better")
                    print("Better")
                else:
                    print("随机接受一些不好的更新")
                    r = np.random.rand()
                    if r < np.exp(-(C["f"] - bestS["f"])/T):                
                        bestS = C
            T *= alpha
            
        num_of_random = num_of_random - 1
    num_of_random = 30
    num_of_evaluation = num_of_evaluation - 1


# In[16]:


bestS


# In[17]:


H


# In[57]:


solu = {'assert': [1, 3, 4, 5, 8, 12, 13, 18, 20, 28], 'f': -0.0031753006752309638,
        'prop': np.array([11, 77, 87,  1, 24, 41,  7, 11,  0, 44])}
renda = 0.5
eval(solu, renda)


# In[58]:


H


# In[59]:


bestS


# In[31]:


assert_items = solu['assert']
s = solu['prop']

F = 1 - omega*num_to_select

L = np.sum(s)  
# F, L, s
print("L:", L, "F:", F, "s：", s)
w = s*F/L + 0.01


# In[32]:


w

