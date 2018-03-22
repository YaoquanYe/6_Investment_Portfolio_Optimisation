
# coding: utf-8

# In[6]:


import numpy as np
import copy

import matplotlib  
import matplotlib.pyplot as plt  
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


# In[19]:


num_of_evaluation = 100       # assert_num*100
num_of_random = 30            # 30
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

E = 50
alpha = 0.95
init_times = 1000
T = 0
T_star = 10  # 10
iterNum_of_T = 10 # 10

to_plot_allE = {}
to_plot_CoVar = [100000]
to_plot_R = [0]


# In[ ]:


for e in range(1, E):
    renda = (e-1)/(E-1)
    num_of_evaluation = 100       # assert_num*100
    num_of_random = 30            # 30
    
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

    # alpha = 0.95
    init_times = 50  # 1000
    T = 0
    
    to_plot_CoVar = [100000]
    to_plot_R = [0]
    
    while num_of_evaluation:# init
        while num_of_random:
            np.random.seed()

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

            T = abs(bestS["f"]/10)
            for i in range(T_star):  # 
                for j in range(iterNum_of_T):  # 
                    print("T_star:" , T_star)
                    print("e:", e, "num_of_evaluation:", num_of_evaluation, "num_of_random:", num_of_random, "SA No:", i, " ", j)

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
#                     print("f of bestS:", bestS["f"])

                    if C["f"] <= bestS['f']:
                        # 已经在eval里面更新了bestS
#                         print("Better")
#                         print("Better")
#                         print("Better")
                        pass
                    else:
#                         print("随机接受一些不好的更新")
                        r = np.random.rand()
                        if r < np.exp(-(C["f"] - bestS["f"])/T):                
                            bestS = C

                    
                    F = 1 - omega*num_to_select
                    L = np.sum(C["prop"])    
                    w = C["prop"]*F/L + 0.01
                    prop = np.zeros(assert_num)
                    index = 0
                    for iii in range(assert_num):
                        if (iii in C["assert"]):
                            prop[iii] = w[index]
                            index += 1
                        else:
                            prop[iii] = 0
                    
                    R = np.sum(prop * RetAndDev[:, 0])
                    #     print("R:", R)
                    CoVar = 0
                    for ii in range(assert_num):
                        for jj in range(assert_num):
                            tmp = prop[ii] * prop[jj] * correlation[ii, jj] * RetAndDev[ii, 0] * RetAndDev[ii, 1]
                            CoVar += tmp

                    if R > to_plot_R[-1] and CoVar < to_plot_CoVar[-1]:
                        print("Add to to_plot")
                        to_plot_R.append(R)
                        to_plot_CoVar.append(CoVar)
                    
                T *= alpha

            num_of_random = num_of_random - 1
        num_of_random = 30  ###############
        num_of_evaluation = num_of_evaluation - 1
    
    to_plot_allE[e] = {"R": to_plot_R, "CoVar": to_plot_CoVar}
np.save("Q4_to_plot_allE.npy", to_plot_allE)


# In[10]:


to_plot_allE


# In[17]:


for i in to_plot_allE:
    print(i)
   
    f1 = plt.figure(1)  
#     plt.subplot(211)  
    plt.scatter(to_plot_allE[i]["CoVar"][1:], to_plot_allE[i]["R"][1:])
    plt.ylabel("R")  
    plt.xlabel("CoVar")
    plt.ylim((0.002, 0.007))
    plt.xlim((0, 0.001))
    plt.show()


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

