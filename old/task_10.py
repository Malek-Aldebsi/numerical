import numpy as np
import sympy
import matplotlib.pyplot as plt
import math
import random
import scipy

from sympy import diff, symbols, factorial, exp, pretty_print, sin, lambdify, pprint
from scipy.linalg import lu, lu_factor, lu_solve

values = {"a11": 3,
          "a12": 1,
          'a13': -4,
          'a14': 1,

          'a21': 1,
          'a22': 3,
          'a23': -.2,
          'a24': 2,

          'a31': .2,
          'a32': -9,
          'a33': 17,
          'a34': 3,

          'a41': 1,
          'a42': 2,
          'a43': 3,
          'a44': 4,

          'b1': -27,
          'b2': 5.4,
          'b3': 118.2,
          'b4': 0,

          's1': 0,
          's2': 0,
          's3': 0,
          's4': 0,

          "x1": 0,
          "x2": 0,
          "x3": 0,
          "x4": 0,

          "xt1": 1,
          "xt2": 2,
          "xt3": 5,
          "xt4": 10,

          "l": 1.22

          }
X1 = []
X2 = []
X3 = []
X4 = []
T1 = []
T2 = []
T3 = []
T4 = []
A1 = ["-"]
A2 = ["-"]
A3 = ["-"]
A4 = ["-"]

x1, x2, x3, x4 = symbols("x1,x2,x3,x4")

B = sympy.Matrix([[-27, 5.4, 118.2]])

func1 = values['a11'] * x1 + values['a12'] * x2 - values['a13'] * x3 + values['a14'] * x4
func2 = values['a21'] * x1 + values['a22'] * x2 + values['a23'] * x3 + values['a24'] * x4
func3 = values['a31'] * x1 + values['a32'] * x2 + values['a33'] * x3 + values['a34'] * x4
fun4 = values['a41'] * x1 + values['a42'] * x2 + values['a43'] * x3 + values['a44'] * x4

fn1 = values['a12'] * x2 + values['a13'] * x3 + values['a14'] * x4
fn2 = values['a21'] * x1 + values['a23'] * x3 + values['a24'] * x4
fn3 = values['a31'] * x1 + values['a32'] * x2 + values['a34'] * x4
fn4 = values['a31'] * x1 + values['a32'] * x2 + values['a33'] * x3

i = 10
# for k in range(i):


#     if(k!=0):
#         if(values['a11']!=0):
#             old1=b1
#         if(values['a12']!=0):
#             old2=b2
#         if(values['a13']!=0):
#             old3=b3
#         if(values['a14']!=0):
#             old4=b4


#     if(values['a11']!=0):
#         b1=( values['b1']-fn1.subs([ (x2, values['s2']), (x3, values['s3']),(x4,values['s4'])]))/values['a11']

#         print("("+"b1"+"-"+str(values['a12'])+"*x2"+"-"+str(values['a13'])+"*x3"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a11")
#         print("("+str(values['b1'])+"-"+str(fn1.subs([ (x2, values['s2']), (x3, values['s3']),(x4,values['s4'])]))+")"+"/"+str(values['a11'])+" =")
#         print('x1: '+str(b1))
#         X1.append(b1)

#     if(values['a12']!=0):
#         b2= (values['b2']-fn2.subs([(x1, values['s1']),  (x3, values['s3']),(x4,values['s4'])]))/values["a22"]
#         print("("+"b2"+"-"+str(values['a21'])+"*x1"+"-"+str(values['a23'])+"*x3"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a22")
#         print("("+str(values['b2'])+"-"+str(fn2.subs([(x1, values['s1']),  (x3, values['s3']),(x4,values['s4'])]))+")"+"/"+str(values["a22"])+" =")
#         print('x2: '+str(b2))
#         X2.append(b2)

#     if(values['a13']!=0):
#        b3= (values['b3']-fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4,values['s4'])]))/values["a33"]
#        print("("+"b3"+"-"+str(values['a31'])+"*x1"+"-"+str(values['a32'])+"*x2"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a33")
#        print("("+str(values['b3'])+"-"+str(fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4,values['s4'])]))+")"+"/"+str(values["a33"])+" =")
#        print('x3: '+str(b3))
#        X3.append(b3)
#        print(str(k)+"--------------------------------------")

#     if(values['a14']!=0):
#         b4=(values['b4']- fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])]))/values["a44"]
#         print("("+"b4"+"-"+str(values['a41'])+"*x1"+"-"+str(values['a42'])+"*x2"+"-"+str(values['a43'])+"*x3"+")"+"/"+"a44")
#         print("("+str(values['b4'])+"-"+str(fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])]))+")"+"/"+str(values["a44"])+" =")
#         print('x4: '+str(b4))
#         X4.append(b4)
#         print(str(k)+"--------------------------------------")

#     if(values['a11']!=0):
#         values['s1']=b1

#     if(values['a12']!=0):
#         values['s2']=b2

#     if(values['a13']!=0):
#         values['s3']=b3

#     if(values['a14']!=0):
#         values['s4']=b4

#     if(values['a11']!=0):
#         t1=abs((values['xt1']-b1)/values['xt1'])*100
#         print("t1 :"+str(t1))
#         T1.append(t1)
#     if(values['a12']!=0):
#         t2=abs((values['xt2']-b2)/values['xt2'])*100
#         print("t2 :"+str(t2))
#         T2.append(t2)
#     if(values['a13']!=0):
#         t3=abs((values['xt3']-b3)/values['xt3'])*100
#         print("t3 :"+str(t3))
#         T3.append(t3)
#         print("------------------------"+str(k))
#     if(values['a14']!=0):
#         t4=abs((values['xt4']-b4)/values['xt4'])*100
#         print("t4 :"+str(t4))
#         T4.append(t4)


#     if(k!=0):
#         if(values['a11']!=0):
#             new1=b1
#         if(values['a12']!=0):
#             new2=b2
#         if(values['a13']!=0):
#             new3=b3
#         if(values['a14']!=0):
#             new4=b4


#         if(values['a11']!=0):
#             a1=abs((new1-old1)/new1)*100
#             print("a1 :"+str(a1))
#             A1.append(a1)
#         if(values['a12']!=0):
#             a2=abs((new2-old2)/new2)*100
#             A2.append(a2)
#             print("a2 :"+str(a2))
#         if(values['a13']!=0):
#             a3=abs((new3-old3)/new3)*100
#             print("a3 :"+str(a3))
#             A3.append(a3)
#             print("--------------------")
#         if(values['a14']!=0):
#             a4=abs((new4-old4)/new4)*100
#             print("a4 :"+str(a4))
#             A4.append(a4)
#             print("--------------------")


#     print(str(k)+"--------------------------------------")
#     print(values['s1'])
#     print(values['s2'])
#     print(values['s3'])


# for k in range(i):
#     if(k!=0):
#         if(values['a11']!=0):
#             old1=b1
#         if(values['a12']!=0):
#             old2=b2
#         if(values['a13']!=0):
#             old3=b3
#         if(values['a14']!=0):
#             old4=b4


#     if(values['a11']!=0):
#         b1=( values['b1']-fn1.subs([ (x2, values['s2']), (x3, values['s3']),(x4,values['s4'])]))/values['a11']
#         values['s1']=b1
#         X1.append(b1)

#         print("("+"b1"+"-"+str(values['a12'])+"*x2"+"-"+str(values['a13'])+"*x3"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a11")
#         print("("+str(values['b1'])+"-"+str(fn1.subs([ (x2, values['s2']), (x3, values['s3']),(x4,values['s4'])]))+")"+"/"+str(values['a11'])+" =")

#         print('b1: '+str(b1))

#     if(values['a12']!=0):
#         b2= (values['b2']-fn2.subs([(x1, values['s1']),  (x3, values['s3']),(x4,values['s4'])]))/values["a22"]
#         values['s2']=b2
#         X2.append(b2)
#         print("("+"b2"+"-"+str(values['a21'])+"*x1"+"-"+str(values['a23'])+"*x3"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a22")
#         print("("+str(values['b2'])+"-"+str(fn2.subs([(x1, values['s1']),  (x3, values['s3']),(x4,values['s4'])]))+")"+"/"+str(values["a22"])+" =")

#         print('b2: '+str(b2))

#     if(values['a13']!=0):
#        b3= (values['b3']-fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4,values['s4'])]))/values["a33"]
#        values['s3']=b3
#        X3.append(b3)
#        print("("+"b3"+"-"+str(values['a31'])+"*x1"+"-"+str(values['a32'])+"*x2"+"-"+str(values['a14'])+"*x4"+")"+"/"+"a33")
#        print("("+str(values['b3'])+"-"+str(fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4,values['s4'])]))+")"+"/"+str(values["a33"])+" =")

#        print('b3: '+str(b3))
#        print(str(k)+"--------------------------------------")

#     if(values['a14']!=0):
#         b4=(values['b4']- fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])]))/values["a44"]
#         values['s4']=b4
#         X4.append(b4)
#         print("("+"b4"+"-"+str(values['a41'])+"*x1"+"-"+str(values['a42'])+"*x2"+"-"+str(values['a43'])+"*x3"+")"+"/"+"a44")
#         print("("+str(values['b4'])+"-"+str(fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])]))+")"+"/"+str(values["a44"])+" =")

#         print('b4: '+str(b4))
#         print(str(k)+"--------------------------------------")


#     if(values['a11']!=0):
#         t1=(abs(values['xt1']-b1)/values['xt1'])*100
#         T1.append(t1)
#         print("t1 :"+str(t1))
#     if(values['a12']!=0):
#         t2=(abs(values['xt2']-b2)/values['xt2'])*100
#         T2.append(t2)
#         print("t2 :"+str(t2))
#     if(values['a13']!=0):
#         t3=(abs(values['xt3']-b3)/values['xt3'])*100
#         T3.append(t3)
#         print("t3 :"+str(t3))
#         print(str(k)+"------------------------")
#     if(values['a14']!=0):
#         t4=(abs(values['xt4']-b4)/values['xt4'])*100
#         T4.append(t4)
#         print("t4 :"+str(t4))
#         print(str(k)+"------------------------")


#     if(k!=0):
#         if(values['a11']!=0):
#             new1=b1
#         if(values['a12']!=0):
#             new2=b2
#         if(values['a13']!=0):
#             new3=b3
#         if(values['a14']!=0):
#             new4=b4


#         if(values['a11']!=0):
#             a1=abs((new1-old1)/new1)*100
#             A1.append(a1)
#             print("a1 :"+str(a1))
#         if(values['a12']!=0):
#             a2=abs((new2-old2)/new2)*100
#             A2.append(a2)
#             print("a2 :"+str(a2))
#         if(values['a13']!=0):
#             a3=abs((new3-old3)/new3)*100
#             A3.append(a3)
#             print("a3 :"+str(a3))
#             print(str(k)+"--------------------")
#         if(values['a14']!=0):
#             a4=abs((new4-old4)/new4)*100
#             A4.append(a4)
#             print("a4 :"+str(a4))
#             print(str(k)+"--------------------")


for k in range(i):
    z = 0

    if (k != 0):
        if (values['a11'] != 0):
            old1 = values['x1']
        if (values['a12'] != 0):
            old2 = values['x2']
        if (values['a13'] != 0):
            old3 = values['x3']
        if (values['a14'] != 0):
            old4 = values['x4']

    if (values['a11'] != 0):
        # print('x1: '+str(values['x1']))
        # print('x1: '+str(values['x2']))
        # print('x1: '+str(values['x3']))
        # print(str(k)+"--------------------------------------")
        b1 = (values['b1'] - fn1.subs([(x2, values['x2']), (x3, values['x3']), (x4, values['x4'])])) / values['a11']
        print("(" + "b1" + "-" + str(values['a12']) + "*x2" + "-" + str(values['a13']) + "*x3" + "-" + str(
            values['a14']) + "*x4" + ")" + "/" + "a11")
        print("(" + str(values['b1']) + "-" + str(
            fn1.subs([(x2, values['s2']), (x3, values['s3']), (x4, values['s4'])])) + ")" + "/" + str(
            values['a11']) + " =")
        print("X1gs= " + str(b1))

        values['s1'] = b1
        # print('b1: '+str(values['s1']))
        c1 = (values['l'] * values['s1']) + ((1 - values['l']) * values['x1'])
        print(str(values['l']) + "*(" + str(values['s1']) + ")" + "+" + "(1-" + str(values['l']) + ")" + "*(" + str(
            values['x1']) + ")" + "= ")
        values['x1'] = c1
        X1.append(c1)

        print('x1: ' + str(values['x1']))
        print("----------------")

    if (values['a12'] != 0):
        b2 = (values['b2'] - fn2.subs([(x1, values['x1']), (x3, values['x3']), (x4, values['x4'])])) / values["a22"]
        print("(" + "b2" + "-" + str(values['a21']) + "*x1" + "-" + str(values['a23']) + "*x3" + "-" + str(
            values['a14']) + "*x4" + ")" + "/" + "a22")
        print("(" + str(values['b2']) + "-" + str(
            fn2.subs([(x1, values['s1']), (x3, values['s3']), (x4, values['s4'])])) + ")" + "/" + str(
            values["a22"]) + " =")
        print("X2gs= " + str(b2))

        values['s2'] = b2
        c2 = values['l'] * values['s2'] + (1 - values['l']) * values['x2']
        print(str(values['l']) + "*(" + str(values['s2']) + ")" + "+" + "(1-" + str(values['l']) + ")" + "*(" + str(
            values['x2']) + ")" + "= ")
        values['x2'] = c2
        X2.append(c2)
        print('x2: ' + str(values['x2']))
        print("----------------")

    if (values['a13'] != 0):
        b3 = (values['b3'] - fn3.subs([(x1, values['x1']), (x2, values['x2']), (x4, values['x4'])])) / values["a33"]
        print("(" + "b3" + "-" + str(values['a31']) + "*x1" + "-" + str(values['a32']) + "*x2" + "-" + str(
            values['a14']) + "*x4" + ")" + "/" + "a33")
        print("(" + str(values['b3']) + "-" + str(
            fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4, values['s4'])])) + ")" + "/" + str(
            values["a33"]) + " =")
        print("X3gs= " + str(b2))

        values['s3'] = b3
        c3 = values['l'] * values['s3'] + (1 - values['l']) * values['x3']

        print(str(values['l']) + "*(" + str(values['s3']) + ")" + "+" + "(1-" + str(values['l']) + ")" + "*(" + str(
            values['x3']) + ")" + "= ")
        values['x3'] = c3
        X3.append(c3)
        print("----------------")

        print('x3: ' + str(values['x3']))
        print(str(k) + "########################################################################")

    if (values['a14'] != 0):
        b4 = (values['b4'] - fn4.subs([(x1, values['x1']), (x2, values['x2']), (x3, values['x3'])])) / values["a44"]
        print("(" + "b4" + "-" + str(values['a41']) + "*x1" + "-" + str(values['a42']) + "*x2" + "-" + str(
            values['a43']) + "*x3" + ")" + "/" + "a44")
        print("(" + str(values['b4']) + "-" + str(
            fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])])) + ")" + "/" + str(
            values["a44"]) + " =")
        print("X4gs= " + str(b4))
        print("----------------")
        values['s4'] = b4
        c4 = values['l'] * values['s4'] + (1 - values['l']) * values['x4']
        print(str(values['l']) + "*(" + str(values['s4']) + ")" + "+" + "(1-" + str(values['l']) + ")" + "*(" + str(
            values['x4']) + ")" + "= ")
        values['x4'] = c4
        X4.append(c4)
        print('x3: ' + str(values['x3']))
        print(str(k) + "########################################################################")

    if (values['a11'] != 0):
        t1 = abs((values['xt1'] - values['x1']) / values['xt1']) * 100
        print("t1 :" + str(t1))
        T1.append(t1)
    if (values['a12'] != 0):
        t2 = abs((values['xt2'] - values['x2']) / values['xt2']) * 100
        print("t2 :" + str(t2))
        T2.append(t2)
    if (values['a13'] != 0):
        t3 = abs((values['xt3'] - values['x3']) / values['xt3']) * 100
        print("t3 :" + str(t3))
        T3.append(t3)
    if (values['a14'] != 0):
        t4 = (abs(values['xt4'] - values['x4']) / values['xt4']) * 100
        print("t4 :" + str(t4))
        T4.append(t4)

    # print(t1)
    # print('x1: '+str(values['x1']))
    # print('xt1: '+str(values['xt1']))
    # print(t2)
    # print('x2: '+str(values['x2']))
    # print('xt2: '+str(values['xt2']))
    # print(t3)
    # print('x3: '+str(values['x3']))
    # print('xt3: '+str(values['xt3']))
    # print(str(k)+"--------------------------------------")

    if (k != 0):
        if (values['a11'] != 0):
            new1 = values['x1']
        if (values['a12'] != 0):
            new2 = values['x2']
        if (values['a13'] != 0):
            new3 = values['x3']
        if (values['a14'] != 0):
            new4 = values['x4']

        if (values['a11'] != 0):
            a1 = abs((new1 - old1) / new1) * 100
            print("a1 :" + str(a1))
            A1.append(a1)
        if (values['a12'] != 0):
            a2 = abs((new2 - old2) / new2) * 100
            print("a2 :" + str(a2))
            A2.append(a2)
        if (values['a13'] != 0):
            a3 = abs((new3 - old3) / new3) * 100
            A3.append(a3)
            print("a3 :" + str(a3))
            print(str(k) + "########################################################################")
        if (values['a14'] != 0):
            a4 = (abs(new4 - old4) / new4) * 100
            A4.append(a4)
            print("a4 :" + str(a4))
            print(str(k) + "########################################################################")
    z = +1

print(A1)
print(A2)
print(A3)
print(A4)
print(X1)
print(X2)
print(X3)
print(X4)
print(T1)
print(T2)
print(T3)
print(T4)

if (values['a14'] == 0):
    A4 = ["-"] * i
    T4 = ["-"] * i
    X4 = ["-"] * i

headerX = "%-20s%-20s%-20s%-20s" % ("x1", "X2", "X3", "X4")
headerA = "%-20s%-20s%-20s%-20s" % ("A1", "A2", "A3", "A4")
headerT = "%-20s%-20s%-20s%-20s" % ("T1", "T2", "T3", "T4")
str_coff_X = ''
str_coff_T = ''
str_coff_A = ''
for n in range(i):
    str_coff_X += "%-20s%-20s%-20s%-20s\n" % (X1[n], X2[n], X3[n], X4[n],)
    str_coff_T += "%-20s%-20s%-20s%-20s\n" % (T1[n], T2[n], T3[n], T4[n],)
    str_coff_A += "%-20s%-20s%-20s%-20s\n" % (A1[n], A2[n], A3[n], A4[n],)

print('------------------------------------')
print(headerX)
print(str_coff_X)
print(headerA)
print(str_coff_A)
print(headerT)
print(str_coff_T)

plt.plot([error for error in range(len(A1[1:]))], [error for error in A1[1:]], label="Et(%) for BiSection")
# plt.plot(false_position_errors.keys(), [error[0] for error in false_position_errors.values()], label="Et(%) for False Position")
# plt.plot(fixed_point_errors.keys(), [error[0] for error in fixed_point_errors.values()], label="Et(%) for Fixed Point")
# plt.plot(newton_raphson_errors.keys(), [error[0] for error in newton_raphson_errors.values()], label="Et(%) for Newton Raphson")
# plt.plot(secant_errors.keys(), [error[0] for error in secant_errors.values()], label="Et(%) for Secant")
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.xlabel('iteration')
plt.ylabel('error(%)')
plt.title('True errors for all method')
plt.show()

# plt.plot(bi_section_errors.keys(), [error[1] for error in bi_section_errors.values()], label="Ea(%) for BiSection")
# plt.plot(false_position_errors.keys(), [error[1] for error in false_position_errors.values()], label="Ea(%) for False Position")
# plt.plot(fixed_point_errors.keys(), [error[1] for error in fixed_point_errors.values()], label="Ea(%) for Fixed Point")
# plt.plot(newton_raphson_errors.keys(), [error[1] for error in newton_raphson_errors.values()], label="Ea(%) for Newton Raphson")
# plt.plot(secant_errors.keys(), [error[1] for error in secant_errors.values()], label="Ea(%) for Secant")
# plt.legend()
# plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
# plt.xlabel('iteration')
# plt.ylabel('error(%)')
# plt.title('Approximate errors for all method')
# plt.show()


# print(str(k)+"--------------------------------------")