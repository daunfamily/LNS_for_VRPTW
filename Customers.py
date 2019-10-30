import numpy as np
import random
import matplotlib.pyplot as plt
CAPCITY = 0
customers = []
dis_matrix = None


# %%

class Route:
    def __init__(self):
        self.c_list = [0, 0]
        self.dis = 0
        self.cap = 0
        self.cur_time = 0

    def __deepcopy__(self):
        new_route = Route()
        new_route.c_list = [i for i in self.c_list]
        new_route.dis = self.dis
        return new_route

    def set_dis(self):
        self.dis = 0
        for i in range(len(self.c_list[:-1])):
            self.dis += dis_matrix[self.c_list[i], self.c_list[i + 1]]

    def set_cap(self):
        self.cap = 0
        for i in self.c_list:
            self.cap += customers[i].cap

    def if_insert(self, i, j):
        """
        能否在i位置插入j号顾客？
        :param i: 顾客位置
        :param j: 顾客编号
        :return: True or False
        """
        self.route_insert(i, j)
        flag = self.check()
        self.c_list.remove(j)
        return flag

    def if_insert_last(self, i):
        self.route_append(i)
        e_time = self.get_e_time(len(self.c_list)-2)
        l_time = self.get_l_time(len(self.c_list)-2)
        self.c_list.remove(i)
        return (e_time <= l_time) and (self.cap + customers[i].cap < CAPCITY)

    def get_e_time(self, j):
        """
        返回j位置的最早时间
        :param j:
        :return:
        """
        if j is 0:
            return customers[j].r_time

        return max(self.get_e_time(j - 1) + dis_matrix[self.c_list[j], self.c_list[j - 1]] + customers[self.c_list[j-1]].s_time,
                   customers[self.c_list[j]].r_time)

    def get_l_time(self, j):
        """
        返回j位置的最迟时间
        :param j:
        :return:
        """
        if j is len(self.c_list) - 1:  # 已经是最后一个了
            return customers[self.c_list[j]].d_time
        return min(customers[self.c_list[j]].d_time,
                   self.get_l_time(j + 1) - dis_matrix[self.c_list[j] - customers[self.c_list[j+1]].s_time, self.c_list[j]])

    def route_append(self, j):  # route 的插入
        self.c_list.insert(-1, j)

    def route_insert(self, i, j):
        """
        i位置插入j顾客
        :param i:
        :param j:
        :return:
        """
        self.c_list.insert(i+1, j)
        self.cap += customers[j].cap
        self.set_dis()

    def route_remove(self, i):
        # 移除route上i号顾客
        tar = self.c_list[i+1]
        self.c_list.remove(tar)
        self.cap -= customers[i].cap
        self.set_dis()

    def route_size(self): # 返回路径中顾客数
        return len(self.c_list)-2

    def check_c(self):
        cap = 0
        for cus in self.c_list:
            cap += customers[cus].cap
        return cap < CAPCITY

    def check_t(self):
        time = 0
        for tmp in range(len(self.c_list)-1):
            time = max(dis_matrix[self.c_list[tmp], self.c_list[tmp+1]] + time, customers[self.c_list[tmp+1]].r_time)
            if time > customers[self.c_list[tmp+1]].d_time:
                return False
            time += customers[self.c_list[tmp+1]].s_time
        return True

    def check(self):
        return self.check_c() and self.check_t()

    def plot(self):
        x = []
        y = []
        for i in self.c_list:
            x.append(customers[i].x)
            y.append(customers[i].y)
        for i in range(len(x)):
            plt.scatter(x[i], y[i], color='r')
        plt.plot(x, y)


class Solution:
    def __init__(self):
        self.r_list = []
        self.dis = 0

    def __copy__(self):
        new_solution = Solution()
        new_solution.r_list = [i.copy() for i in self.r_list]
        new_solution.dis = self.dis

    def set_dis(self):
        self.dis = 0
        for i in self.r_list:
            i.set_dis()
            self.dis += i.dis

    def print(self):
        for route in self.r_list:
            print(route.c_list)
            print(route.check())

    def get_size(self):
        return sum([route.route_size() for route in self.r_list])

    def __deepcopy__(self): # colon from a existing solution
        new_solution = Solution()
        for route in self.r_list:
            new_route = route.__deepcopy__()
            new_solution.r_list.append(new_route)
        new_solution.dis = self.dis
        return new_solution

    def remove_useless_route(self):
        for route in self.r_list:
            if route.route_size() == 0:
                del route

    def plot(self):
        plt.figure()
        for route in self.r_list:
            route.plot()
        plt.show()

# %%

import math


class Customer:
    """
    the class of customer,store and calulate the dis between two cus
    """

    def __init__(self, c_id, x, y, cap, r_time, d_time, s_time):
        self.c_id = int(c_id)
        self.x = int(x)
        self.y = int(y)
        self.cap = int(cap)
        self.r_time = int(r_time)
        self.d_time = int(d_time)
        self.s_time = int(s_time)

    def __lt__(self, other):
        return self.r_time < other.r_time

    def get_dis(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# %%

import re


def read_file(filename):
    """
    read_file
    :param filename: Solomon
    :return: None
    """
    f = open(filename)
    global CAPCITY, customers
    CAPCITY = int(f.readline())
    lines = f.readlines()
    for line in lines:
        line = re.split('(?s)\s+', line)[1:-1]
        if len(line) is 0:
            break
        c_id, x, y, cap, r_time, d_time, s_time = line
        customer = Customer(c_id, x, y, cap, r_time, d_time, s_time)
        customers.append(customer)


# example:
read_file('R101.txt')


# %%

def get_dis_matrix():
    """"""
    global dis_matrix
    dis_matrix = np.zeros([len(customers), len(customers)])
    for i, f_customer in enumerate(customers):
        for j, s_customer in enumerate(customers):
            dis_matrix[i, j] = f_customer.get_dis(s_customer)


get_dis_matrix()


# %%
# test before
# %%
def get_ini_solution():
    """
    获得初始解
    :return:solution by al
    """
    q = sorted(customers[1:])  # 按时间升序排列：d

    if_new_route = False  # FLAG
    # 获得优先队列：
    route = Route()
    solution = Solution()
    while len(q) is not 0:
        if if_new_route:
            solution.r_list.append(route)
            route = Route()
        if_new_route = True
        for i in range(len(q)):
            if route.if_insert_last(q[i].c_id):  # 假如能插入到末尾：
                route.route_append(q[i].c_id)
                q.remove(q[i])
                route.set_cap()
                if_new_route = False
                break
    if len(route.c_list) is not 0:
        solution.r_list.append(route)
    return solution


def get_relation(i, j, solution):
    d_max = np.max(dis_matrix)
    v = 1
    for route in solution.r_list:
        if i and j in route.c_list:
            v = 0
            break
    return 1/(dis_matrix[i][j]/d_max + v)


class cus_for_remove:

    def __init__(self, c_id, other, solution):
        self.c_id = c_id
        self.related_value = get_relation(self.c_id, other, solution)

    def __lt__(self, other):
        return self.related_value < other.related_value


unrelaxed_customers = []
relaxed_customers = []


def lns_remove(solution): # remove, and every time remove once
    P = 5
    global unrelaxed_customers
    global relaxed_customers

    unrelaxed_customers = []
    relaxed_customers = []
    for route in solution.r_list: # 最开始都在unrelaxed中
        unrelaxed_customers += route.c_list[1:-1]
    r = random.randint(0, len(unrelaxed_customers)-1)
    relaxed_customers.append(unrelaxed_customers[r])
    unrelaxed_customers.remove(unrelaxed_customers[r])
    while len(relaxed_customers) <= P-1:
        r = random.randint(0, len(relaxed_customers)-1)
        rel_cuss = []
        for i in unrelaxed_customers:
            cus = cus_for_remove(i, relaxed_customers[r], solution)
            rel_cuss.append(cus)
        rel_cuss.sort()
        num = np.random.uniform()
        num = num * len(unrelaxed_customers)
        relaxed_customers.append(rel_cuss[int(num)].c_id)
        unrelaxed_customers.remove(rel_cuss[int(num)].c_id)
        cus_to_remove = rel_cuss[int(num)]
        for route in solution.r_list:
            if cus_to_remove.c_id in route.c_list:
                route.c_list.remove(cus_to_remove.c_id)
                break

        for route in solution.r_list:
            if relaxed_customers[0] in route.c_list:
                route.c_list.remove(relaxed_customers[0])
                break
    return solution


class Pos:
    def __init__(self, route, post):
        self.route_num = route
        self.pos = post


def find_best_pos(solution, j):
    """
    find j in solution
    :param solution:
    :param j:
    :return:
    """
    ans = 100000
    pos = None

    for tmp, route in enumerate(solution.r_list):
        for i in range(0, route.route_size()+1):
            route.set_dis()
            old_dis = route.dis
            route.route_insert(i, j)
            if route.check():
                if ans > route.dis - old_dis:
                    ans = route.dis - old_dis
                    pos = Pos(tmp, i)
            route.c_list.remove(j)
    return pos


def find_all_pos(solution, j, dis):
    flag = False
    for tmp, route in enumerate(solution.r_list):
        for i in range(0, route.route_size()+1):
            if route.if_insert(i, j):
                route.route_insert(i, j)
                solution.set_dis()
                flag = True
                lns_all_insert(solution, dis)
                route.c_list.remove(j)
    if not flag:
        route = Route()
        route.route_append(j)
        solution.r_list.append(route)
        i = len(solution.r_list)-1
        solution.set_dis()
        lns_all_insert(solution, dis)
        solution.r_list[i].c_list.remove(j)
    relaxed_customers.append(j)


def lns_all_insert(solution,dis):
    solution.set_dis()
    global result_solution
    if solution.dis > dis:
        return
    if len(relaxed_customers) == 0:
        solution.set_dis()
        if solution.dis < result_solution.dis:
            result_solution = solution.__deepcopy__()
    else:
        r = random.randrange(0, len(relaxed_customers))
        c = relaxed_customers[r]
        relaxed_customers.remove(c)
        find_all_pos(solution, c, dis)



def lns_best_insert(solution, dis): # 重新插入过程：

    global result_solution
    if len(relaxed_customers) == 0:
        solution.set_dis()
        result_solution = solution.__deepcopy__()
    else:

        r = random.randrange(0, len(relaxed_customers))
        c = relaxed_customers[r]
        relaxed_customers.remove(c)
        pos = find_best_pos(solution, c)
        if pos is None:
            route = Route()
            route.c_list = [0, c, 0]
            route.set_dis()
            solution.r_list.append(route)
            i = len(solution.r_list) - 1
            j = 0
        else:
            i = pos.route_num
            j = pos.pos
            solution.r_list[i].route_insert(j, c)
        lns_best_insert(solution, dis)
        solution.r_list[i].route_remove(j)
        

solution = get_ini_solution()
solution.print()
solution.plot()
solution.set_dis()
result_solution = solution.__deepcopy__()


for i in range(1000):
    solution = lns_remove(solution)
    lns_all_insert(solution, solution.dis)
    solution = result_solution.__deepcopy__()
    print(result_solution.dis)
    print(result_solution.get_size())

solution.print()
solution.r_list[0].plot()
print(solution.get_size())
solution.plot()

