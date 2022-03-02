--- evaluate.py	(original)
+++ evaluate.py	(refactored)
@@ -10,10 +10,10 @@
 import sys
 import numpy as np
 import itertools
-from __init__ import *
-from submodules import *
-import spconavi_read_data
-import spconavi_save_data
+from .__init__ import *
+from .submodules import *
+from . import spconavi_read_data
+from . import spconavi_save_data
 
 tools     = spconavi_read_data.Tools()
 read_data = spconavi_read_data.ReadingData()
@@ -42,23 +42,23 @@
       
       ##For index of multinominal distribution of place names
       W_index = []
-      for n in xrange(N):
-        for j in xrange(len(Otb[n])):
+      for n in range(N):
+        for j in range(len(Otb[n])):
           if ( (Otb[n][j] in W_index) == False ):
             W_index.append(Otb[n][j])
             #print str(W_index),len(W_index)
       
-      print "[",
-      for i in xrange(len(W_index)):
-        print "\""+ str(i) + ":" + str(W_index[i]) + "\",",
-      print "]"
+      print("[", end=' ')
+      for i in range(len(W_index)):
+        print("\""+ str(i) + ":" + str(W_index[i]) + "\",", end=' ')
+      print("]")
       
       ##Vectorize: Bag-of-Words for each time-step n (=t)
-      Otb_B = [ [0 for i in xrange(len(W_index))] for n in xrange(N) ]
-      
-      for n in xrange(N):
-        for j in xrange(len(Otb[n])):
-          for i in xrange(len(W_index)):
+      Otb_B = [ [0 for i in range(len(W_index))] for n in range(N) ]
+      
+      for n in range(N):
+        for j in range(len(Otb[n])):
+          for i in range(len(W_index)):
             if ( W_index[i] == Otb[n][j] ):
               Otb_B[n][i] += word_increment
       #print Otb_B
@@ -72,13 +72,13 @@
