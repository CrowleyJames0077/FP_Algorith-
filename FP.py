from collections import defaultdict
from email import header
from itertools import combinations, count
from sys import prefix
import pandas as pd
import json
import time
import psutil

path = r"Grocery Products Purchase.csv"

class Node:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None
        
    def increment(self, count):
        self.count += count


def PrepareTransactions(df):
    #code adapted from Pravin Junnarkar. (2020, January 14)
    transactions = []
    for _, row in df.iterrows():
        transaction = set()
        for column in df.columns:
            #code adapted from: W3Schools.com. (2025)
            if pd.notna(row[column]):
                value = str(row[column]).strip()
                if value and value.lower() != 'Null' and value != "":
                    item = f"{value}"
                    transaction.add(item)
        transactions.append(transaction)
    return transactions


#FP growth and tree construction inspired by Chonyy. (2020, October 30)
def FpGrowth(transactionsList, minSupport):
    #print("FP Growth called")
    #code adapted from W3schools. (2018). Python Dictionaries
    itemCounts = defaultdict(int)
    for transaction in transactionsList:
        #print("transaction count")
        for item in transaction:
            #print(f"item: {item}")
            #code adapted from: Python frozenset(). (2025)
            itemCounts[frozenset([item])] += 1
            #print(f"itemCounts: {itemCounts}")
    #print("Item Counts:", dict(itemCounts))

    #filter itemsets
#TODO:: MIn sup Items checks if there are enough counts, rather than if there is enough items in an itemset or items in a transaction
    minSupItems={}
    for itemset,count in itemCounts.items():
        if(count>= minSupport):
        #if (len(itemset) >= minSupport):
            minSupItems[itemset] = count
            #print(f"minSupItems: {minSupItems}")
    #print("Frequent Items:", frequentItems)

    #code adapted from: Python sorted() Function. (n.d.)
    #and also: W3Schools. (2019)
    sortedItems = sorted(minSupItems.items(), key=lambda x: x[1], reverse=True)
    
    #print("counting and filtering done!")
    formattedTransactionSet = []
    frequency = []
    
    for transaction in transactionsList:
        filteredTransaction = []
        for item in transaction:
            if frozenset([item]) in minSupItems:
                filteredTransaction.append(item)
    
        if filteredTransaction:
            filteredTransaction.sort(key=lambda item: minSupItems[frozenset([item])], reverse=True)
            formattedTransactionSet.append(filteredTransaction)
            frequency.append(1)
    
    tree = {}
    headerTable = {}
    
    minSupFormattedTransactionSet = []
    for transaction in formattedTransactionSet:
        #print(f"transaction {transaction} and length {len(transaction)}")
        if(len(transaction) >= minSupport):
            minSupFormattedTransactionSet.append(transaction)
    
    tree, headerTable = ConstructTree(minSupFormattedTransactionSet, frequency)
    newFrequentPatterns= []
    frequentPatterns = []
    level = 0
    #printHeaderTable(headerTable)
    #return MiningTree(headerTable, tree, minSupport)
    frequentPatterns = MiningTree(headerTable, tree, minSupport, newFrequentPatterns)

    return frequentPatterns

                
def ConstructTree(formattedTransactionSet, frequency):
    #print("Tree Constructing")
    tree = {}
    headerTable = defaultdict(int)
    #Explanation for header table:
    #first create header table by adding frequent items
    #then delete any items that are below minSupport
    #create header table columns
    for i, transaction in enumerate(formattedTransactionSet):
        for item in transaction:
            headerTable[item] += frequency[i]


    for item in headerTable:
        headerTable[item] = [headerTable[item],None]
        

    #Now we initialise the header node and the FP Tree.

    tree = Node('Null',1, None)

    for i, transaction in enumerate(formattedTransactionSet):
        #print("building")
        transaction = [item for item in transaction if item in headerTable]
        transaction.sort(key = lambda item: headerTable[item][0], reverse = True)
        currentNode = tree
        for item in transaction:
            #Explanation: adds in tree children and adds in branches to the FP tree
            #if child
            currentNode = updateTree(item, currentNode, headerTable, frequency[i])

    return tree, headerTable


def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        treeNode.children[item].increment(frequency)
        #currentNode.children[item].count += sortedItems[i]['count']
    else:#if parent or is not child
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        updateHeaderTable(item, newItemNode, headerTable)
    return treeNode.children[item]
            
def updateHeaderTable(item, targetNode, headerTable):
    #this now updates the header table, tomake sure it include children nodes as well
    if (headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode
#end of referenced code Chonyy. (2020, October 30)

#code adapted from TÜNEL, M., & GÜLCÜ, ùDEDQ. (2020) Figure 4: Fp-Growth Algorithm Pseudo code
def MiningTree(headerTable, tree,minSupport, frequentItemsets):
    frequentPatterns = []
    #print("Recursive Mining")
    #frequentItemsets = []   
    
    sortedItems = sorted(headerTable.items(), key=lambda x: x[1][0])
    
    isSinglePath = True
    
    currentNode = tree
    #While loop used to find out if tree is a single path
    # If it has more than 1 child, then no its not a single path. else its a single path
    while currentNode:
        # childNodes = list(currentNode.keys())
        childNodes = currentNode.children
        if len(childNodes) > 1: 
            
            isSinglePath = False
            break 
        elif len(childNodes) == 1:
            #currentNode = currentNode[childNodes[0]]['children']
            #currentNode.children[childNodes[0]]
            #code adapted from: Python Dictionary keys() Method. (n.d.)
            isSinglePath = True
            currentNode = childNodes[list(childNodes.keys())[0]]
        else:
            break

    if(isSinglePath):
        #print("is single Path")
        #this particular databsase is not a single path, as far as i know. 
        #TODO: code for single path tree
        #for each subset of items(Y) in tree/singlebranch (B) 
        #output Y union Tree.base where count = smallest count of nodes in Y

                # Traverse the path and generate subsets
        path = []
        currentNode = tree
        #scans the node and its children until there are no more left. This creates the path which will be used for generating subsets
        while currentNode:
            #print("check 1")
            #childNodes = list(currentNode.keys())
            childNodes = currentNode.children
            #print(f"child node lenght {len(childNodes)}")
            if len(childNodes) == 1:
                #Get item and its count
                #item = childNodes[0]
                while childNodes:
                    #print("check 2")
                    child = list(childNodes.values())[0]
                    #print("is this infinte?")
                    path.append((child.name, child.count))
                    childNodes = child.children
                    currentNode = child.children
                    
            else:
                break  #End of the single path

        #generate all subsets of the items
        for i in range(1, len(path) + 1):
            #print("check 3")
            #Generate subsets of size i
            for subset in combinations(path, i):
                #code adapted from: Python - Loop Tuples. (n.d.)
                subsetItems = [x[0] for x in subset]
                subsetCount = [x[1] for x in subset]

                subsetCount = min(subsetCount)
                if (len(subsetItems) >= minSupport):
                    frequentItemsets.append((set(subsetItems), subsetCount))
        #print(f"Single Path Number of Itemsets: {frequentItemsets}")
        #print("not single")
        return frequentItemsets
    else:
        #print("not a single path")
        #if not a single path then for i n Tree.header do:
        #   output Y = Tree.base UNION {i} with i.count
        #if Tree.FP-Array is defined
        #   construct new header table for Ys FPtree from Tree.fp array
        #else construct a new header table from Tree
        #construct Ys conditional Fp-tree Ty and possibly its FP-Array Ay
        #if Ty != 0
        #   call Recursive Mining(tree, 0)
        path=[]
        #goes through each piece of data/node in a header table
        for item, data in headerTable.items():
            #print("check 0")
            newUnionBase = {item}
            supportCount = data[0]
            conditionalPatterns = []
            frequency = []
            conditionalTransactions = []
            node = data[1]
            #keep running until no more nodes to check from. Add node to path, then add the path and nodes-count to conditionalPatterns[]
            while node is not None:
                #print("check 1")
                prefixPath = []
                
                current = node

                while current and current.name != 'Null' and current.name != "Start":
                    prefixPath.append(current.name)
                    current = current.parent
                    
                if prefixPath:
                    #print ("check 3")
                    conditionalPatterns.append((prefixPath, node.count))
                node = node.next
                #if conditional patterns is not empty then construct its own conditional tree and keep mining. 
            #then add the results to subFrequentItems which will form be joined together in frequentItemSets and then returned
            if conditionalPatterns:
                #print("conditional")
                for pattern, count in conditionalPatterns:
                    if(len(pattern) >= minSupport):
                        conditionalTransactions.append(list(pattern))
                        frequency.append(count)

                conditionalTree, conditionalHeader = ConstructTree(conditionalTransactions, frequency)
                #PrintTree(conditionalTree)
                subFrequentItems = MiningTree(conditionalHeader, conditionalTree, minSupport, frequentItemsets)#recursive mining
                #print(f"Not Single Number of SubItemsets: {len(subFrequentItems)}")
                for subItems, subSupport in subFrequentItems:
                    #print("check 4")
                    #code adapted from: Python Set union() Method. (n.d.)
                    extendedItemset = newUnionBase.union(subItems)
                    #print(f"Extended Items : {extendedItemset}")
                    frequentItemSets.append((set(extendedItemset), subSupport))

        return frequentItemsets

     
 
    # 1. Checking if tree is a single path
    # 2. Generating conditional pattern bases
    # 3. Recursively mining conditional trees
    #return frequentItemsets

#end of adapted code TÜNEL, M., & GÜLCÜ, ùDEDQ. (2020) Figure 4: Fp-Growth Algorithm Pseudo code

def FormatResults(frequentItemsets):
    #code adapted from: GeeksforGeeks. (2019, March)
    numbTransactions = len(transactions)
    results = []
    tempResults =[]
    #for each itemset, sort it in order, then add to results ready to be displayed, if no itemsets return a blank list
    for itemset, count in frequentItemsets:
        #print(f"INdividual Itemset: {itemset}")
        cleanedItemset = sorted(item for item in itemset if item.lower() != 'Null')
        if not cleanedItemset:
            continue
        formattedItems = ', '.join(cleanedItemset)
        tempResults.append(formattedItems)
        #print(f"formattedItems{formattedItems}")

        support = count / numbTransactions
        results.append({
            'support': f"{support:.2%}",
            'count': count,
            'items': formattedItems,
        })
    if results != []:
        return pd.DataFrame(results).sort_values('count', ascending=False)
    else:
        return []
#codea dapted from: GeeksforGeeks. (2024, February)
#df = pd.read_csv(path)
#df = pd.read_csv(path).sample(n=40, random_state=42)


numbRows = int(input("Hello, please enter the number of rows from the dataset you wish to use: "))

minSupport = int(input("Also, please enter the minimum support for items in a transaction: "))

df = pd.read_csv(path, nrows=numbRows)
#code adapted from: Monitoring memory usage of a running Python program. (2021, February 23)
memBefore = psutil.Process().memory_info().rss / (1024 * 1024)
#code adapted from: Python time time() Method. (n.d.)
startTime = time.time()

transactions = PrepareTransactions(df)

#print(f"estimatedTime: {estimatedTime}")
print(f"Number of transactions: {len(transactions)}")
for i, transaction in enumerate(transactions[:5]):  #preview first 5 transactions
    print(f"Transaction {i+1}: {transaction}")
#min_support = 2
frequentItemSets = []

frequentItemSets = FpGrowth(transactions, minSupport)
memAfter = psutil.Process().memory_info().rss / (1024 * 1024)
endTime = time.time()


results = FormatResults(frequentItemSets)

print("\nFormatted Results Preview:")

if results.shape[0] > 0:
    print(results.head(10))
else:
    print("not enough information to form a frequent itemset")
print(f" \n Execution Time: {endTime - startTime:.4f} seconds")

print(f"Memory Usage: {memAfter - memBefore} MB")