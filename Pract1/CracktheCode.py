#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import enet_path
from sklearn.linear_model import lasso_path
from sklearn.linear_model import RBFSampler
# import multiprocessing

Excluded_col = [1041, 1052, 1068, 1089, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1238, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1252, 1253, 1254, 1255, 1256, 1258, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1293, 1294, 1295, 1296, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1346, 1347, 1348, 1349, 1350, 1352, 1354,
                1355, 1356, 1357, 1360, 1362, 1364, 1365, 1366, 1367, 1369, 1370, 1371, 1373, 1374, 1376, 1377, 1380, 1381, 1384, 1385, 1386, 1388, 1389, 1391, 1393, 1395, 1396, 1398, 1399, 1403, 1409, 1412, 1427, 1428, 1431, 1432, 1435, 1437, 1438, 1439, 1440, 1441, 1442, 1444, 1448, 1450, 1451, 1452, 1458, 1459, 1460, 1461, 1464, 1466, 1468, 1474, 1479, 1480, 1491, 1492, 1493, 1494, 1495, 1496, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1510, 1511, 1512, 1514, 1516, 1517, 1518, 1519, 1520, 1522, 1525, 1526, 1527, 1528, 1530, 1531, 1532, 1533, 1537, 1538, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699]

# included_col = [26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 49, 50, 53, 54, 55, 57, 62, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 88, 89, 98, 99, 104, 109, 113, 117, 119, 125, 147, 156, 163, 165, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1218, 1229, 1230, 1233, 1237, 1238, 1240, 1404, 1405, 1407, 1418, 1419, 1425, 1427, 1439, 1460, 1465, 1496, 1513, 1519, 1521, 1526, 1527, 1529, 1531, 1536, 1540, 1543, 1546, 1547, 1550, 1551, 1555, 1558, 1560, 1562, 1565, 1568, 1569, 1570, 1572, 1573, 1574, 1575, 1576, 1578, 1579, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1597, 1598, 1601, 1602, 1604, 1611, 1613, 1614, 1615, 1617, 1621, 1622, 1623, 1624, 1625, 1630, 1631, 1633, 1635, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1665, 1676, 1677, 1681, 1683, 1689, 1691, 1692, 1697, 1702, 1703, 1704, 1707]
# very_reduced = [  19,   48,   56,   74,  137,  168,  174,  206,  306,  334,  359,
#         370,  378,  392,  433,  459,  499,  500,  509,  547,  600,  608,
#         621,  627,  644,  679,  697,  701,  705,  716,  735,  737,  791,
#         828,  849,  891,  908,  944,  963,  973,  981, 1008, 1027, 1036,
#        1084, 1086, 1103, 1118, 1131, 1135, 1137, 1138, 1139, 1148, 1152]

##########################################################################
##########################################################################
##########################################################################
#                                  FOR JUN HAN
##########################################################################
##########################################################################
##########################################################################
# JUNHANNNN !

# COMMENT THE LINES BELLOW (included_col) FOR USING ALL OF THE FEATURES !
# YOU CANN ADD A LIST OF FEATURES TO BE SELECTED (INSTEAD OF THE WHOLE DATASET)
# TO SELECT THE FEATURES YOU CAN USE THE NONZERO FUNCTION:
# clf = M.Lasso(alpha=0.0001)
# print(clf.coef_.nonzero()[0])
# COPY AND PASTE THE LIST THAT THIS GIVES YOU TO THE include_col
# AND THEN COMMENT/UNCOMMENT THE LINES ~104 and ~114

# THEN TO FIT THE DATA AND GENERATE THE RESULTS
# ONCE YOU HAVE LEARNED A MODEL :

# import CracktheCode
# M = CracktheCode.Models('openbabel_rdkit_train.csv', maxrow=190000)
# clf = M.Bayesian()

# import pickle
# f = open('model','w')
# pickle.dump(clf, f)

# RELOAD PYTHON TO SAVE MEMORY

# import CracktheCode
# import pickle
# M = CracktheCode.Models('openbabel_rdkit_train.csv', maxrow=10) #NOT NECESSARY TO LOAD MORE THAN 10 HERE WE DONT FIT ANYMORE
# f = open('model','r')
# clf = pickle.load(f)
# M.Fit(clf, 'openbabel_rdkit_test3.csv', 1000000)

##########################################################################
##########################################################################
##########################################################################
#                                        ENJOY
##########################################################################
##########################################################################
##########################################################################

included_col = [0,   2,   3,   5,   6,   8,  11,  12,  13,  14,  15,  18,  20,
                22,  23,  24,  27,  28,  29,  30,  31,  32,  34,  35,  36,  38,
                43,  45,  48,  49,  50,  51,  52,  53,  54,  55,  57,  60,  62,
                64,  65,  70,  71,  73,  76,  77,  79,  83,  85,  86,  87,  88,
                89,  90,  91,  93,  95,  96,  98,  99, 100, 101, 103, 104, 106,
                108, 110, 113, 115, 116, 117, 118, 119, 121, 124, 125, 126, 127,
                128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 143, 144,
                145, 146, 155, 156, 157, 161, 163, 164, 167, 169, 170, 171, 172,
                173, 175, 176, 178, 181, 182, 183, 185, 187, 188, 189, 190, 192,
                195, 196, 197, 198, 199, 203, 204, 206, 207, 214, 218, 220, 222,
                223, 224, 227, 228, 229, 230, 232, 233, 235, 236, 237, 238, 239,
                240, 242, 243, 245, 246, 247, 250, 251, 252, 253, 254, 257, 259,
                260, 261, 262, 270, 274, 276, 279, 280, 281, 282, 283, 286, 288,
                290, 296, 297, 299, 301, 304, 305, 308, 309, 310, 311, 312, 315,
                316, 317, 318, 319, 320, 321, 322, 323, 327, 331, 333, 336, 337,
                338, 339, 341, 342, 343, 344, 345, 347, 349, 351, 355, 356, 358,
                359, 365, 367, 368, 369, 371, 372, 373, 374, 375, 376, 377, 379,
                380, 383, 384, 385, 386, 387, 388, 389, 391, 393, 394, 396, 397,
                399, 403, 405, 406, 407, 408, 409, 414, 416, 417, 419, 422, 423,
                425, 428, 429, 430, 431, 433, 435, 436, 438, 439, 440, 441, 442,
                443, 444, 445, 448, 449, 451, 452, 454, 456, 459, 460, 461, 462,
                463, 465, 467, 470, 472, 473, 474, 475, 476, 478, 479, 481, 482,
                483, 485, 486, 487, 489, 493, 494, 495, 496, 497, 499, 502, 503,
                504, 505, 506, 508, 509, 510, 511, 513, 514, 515, 518, 519, 520,
                521, 522, 523, 524, 526, 527, 528, 533, 536, 537, 538, 539, 540,
                541, 543, 544, 545, 546, 547, 550, 551, 553, 560, 562, 563, 564,
                566, 567, 569, 570, 572, 573, 574, 575, 578, 580, 581, 582, 583,
                584, 585, 587, 588, 590, 591, 592, 593, 594, 595, 596, 597, 598,
                599, 600, 602, 603, 604, 605, 607, 609, 611, 612, 616, 617, 619,
                621, 623, 624, 625, 626, 627, 629, 631, 632, 633, 636, 641]


class Data:

    def __init__(self, File, maxrow=None, filter_col=True, testing=False):
        csv_filename = File
        smiles = []
        features = []
        gapvalue = []
        if testing:
            ID = []
        f = open(csv_filename, 'r')
        line = True
        if maxrow:
            mrow = maxrow
        else:
            mrow = 10000000
        r = 0
        print mrow
        # line = f.readline()
        while line:
            line = f.readline()
            row = line.strip().split(',')
        # with  as csv_fh:
        # Parse as a CSV file.
        # 	reader = csv.reader(csv_fh)
        # Skip the header line.
        # next(reader, None)
        # Loop over the file.
            # for row in reader:
            # if np.random.random() > 0.5:
            # Store the data.
            if testing:
                # print row
                try:
                    ID.append(int(row[0]))
                    smiles.append(str(row[1]))
                    # UNCOMMENT THE LINE BELLOW FOR USING ALL OF THE FEATURES !
                    # features.append(np.array([self.Filter(row[i+2]) for i in range(len(row[2:]))]))
                    gapvalue.append(float(row[-1]))
                    # UNCOMMENT THE LINE BELLOW FOR USING ONLY SELECTED
                    # FEATURES !
                    features.append(
                        np.array([self.Filter(row[i + 2]) for i in included_col]))
                except:
                    print row
            else:
                try:
                    smiles.append(str(row[0]))
                    # UNCOMMENT THE LINE BELLOW FOR USING ALL OF THE FEATURES !
                    # features.append(np.array([self.Filter(row[i+1]) for i in range(len(row[1:-1]))]))
                    gapvalue.append(float(row[-1]))

                    # UNCOMMENT THE LINE BELLOW FOR USING ONLY SELECTED
                    # FEATURES !
                    features.append(
                        np.array([self.Filter(row[i + 1]) for i in included_col]))
                except:
                    print row
            # features.append(np.array([self.Filter(row[i+1]) for i in included_col], dtype=np.bool))
            r += 1
            if r % 10000 == 0:
                print "loaded %s lines! loading ......." % r
            if r > mrow:
                break
        # Turn the data into numpy arrays.
        if testing:
            self.ID = ID
        self.smiles = smiles
        self.features = np.array(features)
        self.gapvalue = np.array(gapvalue)
        self.N = len(self.smiles)

    def Filter(self, x):
        if x == 'nan' or x == 'inf':
            # print "NAN"
            x = 0
        try:
            x = float(x)
        except:
            print "EXCEPTION", x
            x = 0
        return x

    def Bootstrap(self):
        ids = np.arange(self.N)
        np.random.shuffle(ids)
        halfN = int(self.N * 0.75)
        training = ids[0:halfN]
        testing = ids[halfN:-1]
        return training, testing

    def BootstrapOff(self):
        ids = np.arange(self.N)
        np.random.shuffle(ids)
        halfN = int(self.N * 0.99)
        training = ids[0:halfN]
        testing = ids[halfN:-1]
        return training, testing


class Models:

    "This class contains the different regression models, Lasso, Ridge, etc as function"

    def __init__(self, File, maxrow=None, write_result=True):
        self.write_result = write_result
        self.data = Data(File, maxrow)
        self.Xn_predicted = {}  # Yhat
        self.training, self.testing = self.data.Bootstrap()
        self.X = np.array([self.data.features[i] for i in self.training])
        self.X_test = np.array([self.data.features[i] for i in self.testing])
        self.Y = np.array([self.data.gapvalue[i] for i in self.training])

    def Fit(self, clf, File, nline=1000000):
        print "Loading File ..."
        datatofit = Data(File, nline, testing=True)
        X_topredict = np.array(datatofit.features)
        print "Calculating Yhat"
        Yhat = clf.predict(X_topredict)
        print "Writing Results ..."
        f = open('result.csv', 'w')
        f.write('Id,Prediction\n')
        for i in range(len(X_topredict)):
            f.write('%s,%s\n' % (datatofit.ID[i], Yhat[i]))
        f.close()
        # for i in range(len(Yhat)):
        # 	self.Xn_predicted[self.testing[i]] = Yhat[i]
        # self.Y_tofit = np.array(self.data.gapvalue)

    def bootstrap_off(self):
        self.training, self.testing = self.data.BootstrapOff()
        self.X = np.array([self.data.features[i] for i in self.training])
        self.X_test = np.array([self.data.features[i] for i in self.testing])
        self.Y = np.array([self.data.gapvalue[i] for i in self.training])

    def bootstrap_reset(self):
        self.training, self.testing = self.data.Bootstrap()
        self.X = np.array([self.data.features[i] for i in self.training])
        self.X_test = np.array([self.data.features[i] for i in self.testing])
        self.Y = np.array([self.data.gapvalue[i] for i in self.training])

    def RMSE(self):
        if len(self.Xn_predicted.keys()) != 0:
            RMSE = 0
            for i in self.testing:
                RMSE += (self.Xn_predicted[i] - self.data.gapvalue[i]) ** 2
            RMSE = np.sqrt((1.0 / len(self.testing) * RMSE))
            return RMSE
        else:
            return False

    def rsquare(self):
        SSE = 0
        ymean = []
        for i in self.testing:
            SSE += (self.Xn_predicted[i] - self.data.gapvalue[i]) ** 2
            ymean.append(self.data.gapvalue[i])
        ymean = np.mean(ymean)
        TSS = 0
        for i in self.testing:
            TSS += (self.data.gapvalue[i] - ymean) ** 2
        r2 = 1 - (SSE / TSS)
        return r2

    def lstsqr(self):
        X = np.vstack((self.X.T, np.ones(len(self.X)))).T
        w = np.linalg.lstsq(X, self.Y)
        X_test = np.vstack((self.X_test.T, np.ones(len(self.X_test)))).T
        Yhat = np.dot(X_test, w[0])
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return w[0]

    def Lasso(self, alpha=0.01):
        "Will do a lasso here"
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(self.X, self.Y)
        Yhat = clf.predict(self.X_test)
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return clf

    # def KR_gridsearch(self, train_size=1000):

    # 	kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
    #                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
    #                              "gamma": np.logspace(-2, 2, 5)})
    # kr.fit(self.X[:train_size], self.Y[:train_size])
    # 	Yhat = kr.predict(self.X_test)
    # 	for i in range(len(Yhat)):
    # 		self.Xn_predicted[self.testing[i]] = Yhat[i]
    # 	return kr

    def SVR(self):
        "Will do a lasso here"
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X_features = rbf_feature.fit_transform(self.X)
        X_test_features = rbf_feature.fit_transform(self.X_test)
        clf = svm.SVR()
        clf.fit(X_features, self.Y)
        Yhat = clf.predict(X_test_features)
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return clf

    def SVR_grid_gamma(self):
        C_range = 10.0 ** np.arange(-2, 9)
        gamma_range = 10.0 ** np.arange(-5, 4)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedKFold(y=self.Y, n_folds=3)
        grid = GridSearchCV(svm.SVR(), param_grid=param_grid, cv=cv)
        grid.fit(self.X, self.Y)
        print("The best classifier is: ", grid.best_estimator_)

    def Bayesian(self):
        clf = linear_model.BayesianRidge()
        clf.fit(self.X, self.Y)
        Yhat = clf.predict(self.X_test)
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return clf

    def Ridge(self, alpha=None):
        clf = linear_model.Ridge(alpha=alpha)
        clf.fit(self.X, self.Y)
        Yhat = clf.predict(self.X_test)
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return clf

    def ElasticNet(self, alpha=None):
        clf = linear_model.ElasticNet(alpha=alpha)
        clf.fit(self.X, self.Y)
        Yhat = clf.predict(self.X_test)
        for i in range(len(Yhat)):
            self.Xn_predicted[self.testing[i]] = Yhat[i]
        return clf

    def Plot_Lasso_Path(self, path_lenght=5e-3, alphas=None):
        import matplotlib.pyplot as plt
        print("Computing regularization path using the lasso...")
        alphas_lasso, coefs_lasso, _ = lasso_path(
            self.X, self.Y, path_lenght, fit_intercept=False, alphas=alphas)
        print("Computing regularization path using the positive lasso...")
        alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
            self.X, self.Y, path_lenght, positive=True, fit_intercept=False, alphas=alphas)
        plt.figure()
        ax = plt.gca()
        ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
        l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
        l2 = plt.plot(-np.log10(alphas_positive_lasso), coefs_positive_lasso.T,
                      linestyle='--')
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Lasso and positive Lasso')
        plt.legend(
            (l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
        plt.axis('tight')
        plt.show()

    def Plot_Enet_Path(self, path_lenght=5e-3, alphas=None):
        import matplotlib.pyplot as plt
        print("Computing regularization path using the elastic net...")
        alphas_enet, coefs_enet, _ = enet_path(
            self.X, self.Y, eps=path_lenght, l1_ratio=0.8, fit_intercept=False, alphas=alphas)
        print("Computing regularization path using the positve elastic net...")
        alphas_positive_enet, coefs_positive_enet, _ = enet_path(
            self.X, self.Y, eps=path_lenght, l1_ratio=0.8, positive=True, fit_intercept=False, alphas=alphas)
        plt.figure()
        ax = plt.gca()
        ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
        l1 = plt.plot(-np.log10(alphas_enet), coefs_enet.T)
        l2 = plt.plot(-np.log10(alphas_positive_enet), coefs_positive_enet.T,
                      linestyle='--')
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Elastic-Net and positive Elastic-Net')
        plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
                   loc='lower left')
        plt.axis('tight')
        plt.show()

    def Run_Model(self, Class, Model, queue):
        Class.bootstrap_reset()
        w = Model()
        r = Class.RMSE()
        queue.put((r, w))

    def plot_ic_criterion(self, model, name, color):
        alpha_ = model.alpha_
        alphas_ = model.alphas_
        criterion_ = model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    def Plot_Lasso_LARS_Path(self):

        X = self.X
        y = self.Y

        # normalize data as done by Lars to allow for comparison
        X /= np.sqrt(np.sum(X ** 2, axis=0))

        #######################################################################
        # LassoLarsIC: least angle regression with BIC/AIC criterion

        model_bic = LassoLarsIC(criterion='bic')
        t1 = time.time()
        model_bic.fit(X, y)
        t_bic = time.time() - t1

        model_aic = LassoLarsIC(criterion='aic')
        model_aic.fit(X, y)

        plt.figure()
        self.plot_ic_criterion(model_aic, 'AIC', 'b')
        self.plot_ic_criterion(model_bic, 'BIC', 'r')
        print model_bic.alpha_
        plt.legend()
        plt.title('Information-criterion for model selection (training time %.3fs)'
                  % t_bic)

        #######################################################################
        # LassoCV: coordinate descent

        # Compute paths
        print(
            "Computing regularization path using the coordinate descent lasso...")
        t1 = time.time()
        model = LassoCV(cv=20).fit(X, y)
        t_lasso_cv = time.time() - t1

        # Display results
        m_log_alphas = -np.log10(model.alphas_)

        plt.figure()
        ymin, ymax = 2300, 3800
        plt.plot(m_log_alphas, model.mse_path_, ':')
        plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                    label='alpha: CV estimate')

        plt.legend()

        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: coordinate descent '
                  '(train time: %.2fs)' % t_lasso_cv)
        plt.axis('tight')
        plt.ylim(ymin, ymax)

        #######################################################################
        # LassoLarsCV: least angle regression

        # Compute paths
        print("Computing regularization path using the Lars lasso...")
        t1 = time.time()
        model = LassoLarsCV(cv=20).fit(X, y)
        t_lasso_lars_cv = time.time() - t1

        # Display results
        m_log_alphas = -np.log10(model.cv_alphas_)

        plt.figure()
        plt.plot(m_log_alphas, model.cv_mse_path_, ':')
        plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                    label='alpha CV')
        plt.legend()

        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
                  % t_lasso_lars_cv)
        plt.axis('tight')
        plt.ylim(ymin, ymax)

        plt.show()


# RMSE = {'lasso':[], 'lstsqr':[], 'ridge':[], 'elastic':[], 'bayesian':[]}
# M = CracktheCode.Models('conso.csv', maxrow=100000)
# X_size = [100,500,1000,2000,5000,10000,20000,50000,99999]
# for train_size in X_size:
# 	M.bootstrap_reset()
# 	M.X = M.X[:train_size]
# 	M.Y = M.Y[:train_size]
# 	M.lstsqr()
# 	RMSE['lstsqr'].append(M.RMSE())
# 	M.Lasso(alpha=0.0001)
# 	RMSE['lasso'].append(M.RMSE())
# 	M.Ridge(alpha=0.0001)
# 	RMSE['ridge'].append(M.RMSE())
# 	M.ElasticNet(alpha=0.0001)
# 	RMSE['elastic'].append(M.RMSE())
# 	M.Bayesian()
# 	RMSE['bayesian'].append(M.RMSE())


# fig = plt.figure(figsize=(20,15))
# ax= fig.add_subplot(111)
# ppl.plot(np.log10(X_size), RMSE['lstsqr'],  label='Least-Square')
# ppl.plot(np.log10(X_size), RMSE['lasso'],  label='Lasso')
# ppl.plot(np.log10(X_size), RMSE['ridge'],  label='Ridge')
# ppl.plot(np.log10(X_size), RMSE['elastic'],  label='ElasticNet')
# ppl.plot(np.log10(X_size), RMSE['bayesian'],  label='BayesianRidge')
# ax.set_xlabel('log(Training set size)')
# ax.set_ylabel('RMSE')
# ax.set_ylim((0,1))
# ax.set_title('RMSE as a function of training set size')
# plt.legend()
# fig.savefig('RMSE_VS_size.png')


# RMSE = {'lasso':[], 'elastic':[], 'bayesian':[]}
# files = ['conso.csv','openbabel_train.csv','train.csv']
# for dataset in files:
# 	M = CracktheCode.Models(dataset, maxrow=10001)
# 	M.Lasso(alpha=0.0001)
# 	RMSE['lasso'].append(M.RMSE())
# 	M.ElasticNet(alpha=0.0001)
# 	RMSE['elastic'].append(M.RMSE())
# 	M.Bayesian()
# 	RMSE['bayesian'].append(M.RMSE())

# import prettyplotlib as ppl

# fig = plt.figure(figsize=(20,15))
# ax= fig.add_subplot(131)
# ppl.bar(ax, np.arange(len(RMSE['lasso'])), RMSE['lasso'], annotate=True, xticklabels=['OpenBabel+RDkit','OpenBabel','RDkit'])
# ax.set_ylabel('RMSE')
# ax.set_title('Lasso')
# ax= fig.add_subplot(132)
# ax.set_ylabel('RMSE')
# ax.set_title('Elastic Net')
# ppl.bar(ax, np.arange(len(RMSE['elastic'])), RMSE['elastic'], annotate=True, xticklabels=['OpenBabel+RDkit','OpenBabel','RDkit'])
# ax= fig.add_subplot(133)
# ax.set_ylabel('RMSE')
# ax.set_title('BayesianRidge')
# ppl.bar(ax, np.arange(len(RMSE['bayesian'])), RMSE['bayesian'], annotate=True, xticklabels=['OpenBabel+RDkit','OpenBabel','RDkit'])
# fig.savefig('RMSE_VS_feature.png')


# class Run_Multicore:
# 	def __init__(self, pool_size=2):
# 		self.pool_size = pool_size

# 	def CrossValidation(self,Model_class, Method, NB_cv=64):
# 		q = multiprocessing.Queue()
# 		P = []
# 		jobs = 0
# 		res = []
# 		while jobs <= NB_cv:
# 			for i in range(self.pool_size):
# 				p = multiprocessing.Process(target=Model_class.Run_Model, args=(Model_class,Method,q))
# 				p.start()
# 				P.append(p)
# 				jobs += 1
# 			for i in range(self.pool_size):
# 				res.append(q.get())
# 		rs = []
# 		ws = []
# 		for i in res:
# 			rs.append(i[0])
# 			ws.append(i[1])
# 		print rs
# 		return rs, ws


# Example Code to run the Model Selection:
# import CracktheCode
# M = CracktheCode.Models('train.csv', maxrow=10000)
# w_lasso = M.Lasso(alpha=0.01)
# RMSE_lasso = M.RMSE()
# w_lstsqr = M.lstsqr()
# RMSE_lstsqr = M.RMSE()
# print "RMSE is for Lasso: %s and for Least Square: %s"%(RMSE_lasso,
# RMSE_lstsqr)
