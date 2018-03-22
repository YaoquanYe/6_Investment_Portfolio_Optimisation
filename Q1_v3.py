import numpy as np

# Read file
file=open("./datasets/assets1.txt") 
lines=file.readlines() 

# Read the expected return and standard deviation
assert_num = np.int(lines[0].strip().split(' ')[0])
RetAndDev = np.ones([assert_num, 2])
for i in range(1, assert_num+1):
    line = lines[i]
    line=line.strip().split(' ')
    ret = np.float64(line[0], dtype=np.float)
    stddev = np.float64(line[1], dtype=np.float)
    RetAndDev[i-1, 0] = ret
    RetAndDev[i-1, 1] = stddev
    
# Read correlation
correlation = np.ones([assert_num, assert_num])
for i in range(assert_num+1, assert_num+1+int((1+assert_num)*assert_num/2)):
    line = lines[i]
    line=line.strip().split(' ')
    correlation[np.int(line[0])-1, np.int(line[1])-1] = np.float64(line[2])
    correlation[np.int(line[1])-1, np.int(line[0])-1] = np.float64(line[2])

# Set the initial variables
num_of_evaluation = assert_num*1000 # You can change the number of evalutions here to let the program run faster
num_of_random = 30
num_to_select = 10
renda = 0.5
omega = 0.01
delta = 0.91 #the maximum proportion that can be held of asset i
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

# Maximum number of function evaluations
while num_of_evaluation:# init
    
    # Determine the solution is infeasible or not
    if num_to_select*omega > 1 or num_to_select*delta < 1: ##new
        break

    # Random generate a solution Q set for this program to determine
    # if it is the best set. All the selected asset is marked as integer 1
    # in the array.
    items_tmp = np.zeros([assert_num])
    items_tmp[:num_to_select] = 1
    np.random.shuffle(items_tmp)
    Q = []
    for i in range(assert_num):
        if items_tmp[i] == 1:
            Q.append(i)
    print("Q:", Q)
    
    #Set variable F
    F = 1 - omega*num_to_select
    print("F:", F)
    
    #Set the returned objective function value to maximum number
    f = 100000000000

    # Repeat each run with a different initial random seed 30 times
    while num_of_random:
        
        # Set seed
        np.random.seed(num_of_random)
        print(num_of_evaluation, num_of_random) # "evalNO randomNo:",
        
        # Set the current value for the selected asset and sum them up
        # to calculate L and w
        s = np.random.randint(1, size=num_to_select)
        L = np.sum(s)
        w = s*F/L + 0.01
        
        # Set a list called prop to store the proportion
        # associated with asset in the current solution
        index = 0
        prop = np.zeros(assert_num)
        
        for i in range(assert_num):
            if (i in Q):
                prop[i] = w[index]
                index += 1
            else:
                prop[i] = 0
        
        # Calculate the expected return of the current solution
        R = np.sum(prop * RetAndDev[:, 0])
        print("R:", R)
        
        # Calculate the expected variance of the portfolio
        CoVar = 0
        for i in range(assert_num):
            for j in range(assert_num):
                tmp = prop[i] * prop[j] * correlation[i, j] * RetAndDev[i, 1] * RetAndDev[j, 1]
                CoVar += tmp
        print("CoVar:", CoVar)
        
        # Calculate the returned objective function value for the current solution
        f = renda*CoVar - (1-renda)*R
        print("R:", R)
        
        # Record the the expected return of the current solution
        # and the the expected variance of the portfolio
        R_list.append(R)
        CoVar_list.append(CoVar)
        
        # Update the worse returned objective function value if f > max_f
        # Record just for record
        if f > max_f:
            max_f = f
            max_R = R
            max_CoVar = CoVar
        
        # Update the better returned objective function value if f < min_f
        # Put it into a improved solution list and update the the best solution set
        if f < min_f:
            min_f = f
            min_R = R
            min_CoVar = CoVar
            H.append({"assert": Q, "prop": s, "f":f})
            bestS = {"assert": Q, "prop": s, "f":f}
        
        #  Decrement the num_of_random to try a different seed
        num_of_random = num_of_random - 1
    
    # Decrement the number of evaluation
    num_of_random = 30
    num_of_evaluation = num_of_evaluation - 1


print("The best solution is:", bestS)

print("The improved solution set is:", H)


# In[5]:


R_list = np.array(R_list)
CoVar_list = np.array(CoVar_list)


# In[6]:


R_list.max(), R_list.min(), R_list.mean(), R_list.std()


# In[8]:


CoVar_list.max(), CoVar_list.min(), CoVar_list.mean(), CoVar_list.std()

