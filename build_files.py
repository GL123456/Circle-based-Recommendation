#!/bin/python

import sys
import os
import subprocess

# This script should be run from the folder ./epinions/epinions/epinions/
 # for creating required files

user_l_limit = 0
user_u_limit = 2000

user_list = []   # list of all the users
reviews = {}     # key : userID ,val : list of lists, each list contain values (producttext, categorytext, productrating, reviewrating)
pc_map = {}      # key : product, val : category
pid_map = {}     # key : producttext, val : product_id
cid_map = {}     # key : categorytext, val : category_id
trusts = {}      # key : userID ,val : list of all other users
trustedby = {}   # key : userID ,val : list of other users

# Extract all the userID present in data

def remove_spaces(s):  # replace spaces by underscore
    s = s.split(" ")
    s = "_".join(s)
    return s

ptop = 1
ctop = 1

user_list = subprocess.check_output("ls userpages/", shell = True)
user_list = user_list.split("\n");
for i,user in enumerate(user_list):
    user_list[i] = user[:-5]
user_list.remove('')
print "List of Users extracted"

# For extracting reviews of all the users to reviews dictionary
for user in user_list:
    path = "./reviews/" + user
    lst = []
    temp_review_data = open(path, "r+").readlines()
    for line in temp_review_data:
        line = line.split("\t")
        producttext = line[3]
        producttext = remove_spaces(producttext)
        if producttext not in pid_map:
            pid_map[producttext] = ptop
            ptop += 1
        categorytext = line[5]
        categorytext = remove_spaces(categorytext)
        if categorytext not in cid_map:
            cid_map[categorytext] = ctop
            ctop += 1
        productrating = line[7]
        reviewrating = line[8]
        pc_map[pid_map[producttext]] = categorytext

        temp_list = []
        temp_list.append(pid_map[producttext])
        temp_list.append(cid_map[categorytext])
        temp_list.append(productrating)  # It can be 'na' also
        temp_list.append(reviewrating)
        lst.append(temp_list)
    reviews[user] = lst

print "All reviews extracted"

#For extracting followers and following
for user in user_list:
    path = "./wot/" + user
    temp_trusts = []
    temp_trustedby = []
    path_trusts = path + "-trusts"
    path_trustedby = path + "-trustedby"
    data_trusts = open(path_trusts, "r+").readlines()
    data_trustedby = open(path_trustedby, "r+").readlines()
    for temp in data_trusts:
        temp = temp.split("\t")[0]
        temp_trusts.append(temp)
    for temp in data_trustedby:
        temp = temp.split("\t")[0]
        temp_trustedby.append(temp)
    trusts[user] = temp_trusts
    trustedby[user] = temp_trustedby

print "All the trusts and trustedby values extracted"

print "Creating User_Trust_Network.csv"
# For creating User_Trust_Network.txt
fp = open("User_Trust_Network.csv", "w")
fp.write("User_id:Trustee\n")
for user in user_list:
    if int(user) >= user_l_limit and int(user) < user_u_limit:
        lst = trusts[user]
        s = ""
        for i in lst:
            if int(i) >= user_l_limit and int(i) < user_u_limit:
                s += str(i) + ","
        s = s[:-1]
        if len(s)!=0:
            fp.write(user + ":" + s + "\n")
fp.close()
print "User_Trust_Network.txt created\n"

print "creating Category.txt"
fp = open("Category.txt", "w")
for c in cid_map:
    fp.write(c + "\n")
fp.close()
print "created Category.txt\n"

print "creating Products.csv"
fp = open("Products.csv", "w")
fp.write("Product_id,Category_names\n")
for pr in pc_map:
    fp.write(str(pr) + "," + str(pc_map[pr]) + "\n")
fp.close()
print "created Products.csv\n"

print "creating Rating.csv"
fp = open("Rating.csv", "w")
fp.write("User_id,Product_id,Rating\n")
for user in reviews:
    if int(user) >= user_l_limit and int(user) < user_u_limit:
        for r in reviews[user]:
            s = ""
            s += user + "," + str(r[0]) + "," + r[3]
            fp.write(s + "\n")
fp.close()
print "created Rating.txt"

print "creating product_productIDmap.txt"
fp = open("Product_ProductIDmap.txt", "w")
fp.write("Product_id,Producttext\n")
for product in pid_map:
    fp.write(str(pid_map[product]) + " " + product + "\n")
fp.close()
print "created product_productIDmap.txt\n"
