loss_threshold = 0
shuffle_dataset = True
train_data_ratio = 0.8
learning_rate = 0.01
delta = 0.01 #Percentage
viz = True

def linear_congruential_generator(seed):
#https://en.wikipedia.org/wiki/Linear_congruential_generator
    def rand(precision = 5):
        nonlocal seed
        seed = (1103353515245*seed + 1234567) & 0x7fffffff
        return (seed%(10**precision))/(10**precision)
    return rand

rnd = linear_congruential_generator(1)

def read_csv(fname):
    data = []
    catagoricals_index = set()
    with open(fname, 'r') as f:
        num_features = len(f.readline().strip().split(','))
        for row in f:
            cur_row = [1]
            for el in row.strip().split(','):
                try:
                    cur_row.append(float(el))
                except ValueError:
                    catagoricals_index.add(len(cur_row))
                    cur_row.append(el)
            data.append(cur_row)
    return data, list(catagoricals_index), num_features

data, catagoricals_index, num_features = read_csv('Salary.csv')
#https://www.kaggle.com/rsadiq/salary

if catagoricals_index:
    transpose = list(map(list, zip(*data)))
    for i in sorted(catagoricals_index, reverse = True):
        v = list(set(transpose[i]))
        num_features += len(v)-2
        #-2 because we are removing the actual column and adding number_of_distinct_values-1 columns
        mapper = {v[k]:[1 if j==k else 0 for j in range(len(v)-1)] for k in range(len(v))}
        for row in range(len(data)):
            data[row].extend(mapper[data[row][i]]+[data[row].pop()])
            data[row].pop(i)

if shuffle_dataset:
    new_data = []
    digits = len(str(len(data)))
    while len(data):
        k = int((10**2)*rnd(digits))
        try: new_data.append(data.pop(k))
        except IndexError: pass
    data = new_data

train = data[0:int(len(data)*train_data_ratio)]
test = data[len(train):]
viz = viz and num_features == 2

def loss_calc(weights, data):
    assert len(weights) == len(data[0])-1
    loss = 0
    for example in data:
        x = example[:-1]
        y = example[-1]
        loss += (y - sum([w*d for w,d in zip(weights, x)]))**2
    loss /= (2*len(data))
    return loss

weights = [rnd() for x in range(num_features)]
old_loss = loss_calc(weights, train)
transpose = list(map(list, zip(*train)))
delta = [(float(sum(f))/len(f))*delta for f in transpose[:-1]]
itr = 0

if viz:
    init_line = [sum([w*d for w,d in zip(weights, x)]) for x in train]
    intermediate_sols = []        

while True:
    itr+=1
    old_weights = weights.copy()
    for i in range(num_features):
        old_loss = loss_calc(weights, train)
        tmp_weights = weights.copy()
        tmp_weights[i] += delta[i]
        new_loss = loss_calc(tmp_weights, train)
        slope = (new_loss-old_loss)/delta[i]
        weights[i] -= slope*learning_rate

    old_loss = loss_calc(old_weights, train)
    new_loss = loss_calc(weights, train)
    print('Iteration: '+str(itr)+'. Improvement :'+str(abs(old_loss-new_loss)))

    if abs(old_loss-new_loss) <= loss_threshold:
        break
    
    if (viz) and not (itr%100):
        intermediate_sols.append([sum([w*d for w,d in zip(weights, x)]) for x in train])

#Testing
vals = [x[-1] for x in test]
pred = [sum([w*d for w,d in zip(weights, x)]) for x in test]
yavg = sum(vals)/len(vals)
r2 = 1-(sum([(y-p)**2 for y,p in zip(vals,pred)])/sum([(v-yavg)**2 for v in vals]))

if viz:
    import matplotlib.pyplot as plt
    plt.plot(transpose[1], init_line,color = 'r', label='Initial Solution')
    for sol in intermediate_sols:
        plt.plot(transpose[1], sol, color='0.1', linestyle='dashed', alpha=0.1)
    plt.plot(transpose[1],
             [sum([w*d for w,d in zip(weights, x)]) for x in train],
             color = 'b', label='Final Solution')
    plt.scatter(transpose[1], transpose[2], color = 'g', label='Training Data')
    plt.scatter([i[1] for i in test], vals, color = 'm', label='Testing Data')
    plt.legend()
    plt.show()
print('Test Set R Squared Value: '+str(r2))
