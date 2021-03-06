xgb_json = bst.get_dump(dump_format='json')

with codecs.open("E:\Python\model_XGB_json.json", 'w', encoding="utf-8") as f:
   for single_json in xgb_json:
      single_json = single_json.replace('\n',' ').replace('\r', ' ')
      f.write(single_json + '\n')

## 解析单棵数

def parse_xgb_tree_2sql(xgb_tree_json, mid_sqls, tree_num, depth=0):
   indent = " " * (depth+1)
   ## 用于对齐sql语句
        if 'leaf' in xgb_tree_json.keys():
           leaf_value = xgb_tree_json['leaf']

           if(len(mid_sqls)>=1 and 'else' in mid_sqls[-1]):
              cur_sql = indent + str(leaf_value) + ' '
              else:
              cur_sql = indent + str(leaf_value)
              mid_sqls.append(cur_sql)
              return
feat = xgb_tree_json['split']

value = str(xgb_tree_json['split_condition'])

left_tree = xgb_tree_json['yes']

right_tree = xgb_tree_json['no']

missing = xgb_tree_json['missing']

if missing == left_tree:
   cur_sql = '(' + feat + ' is null' + ' or ' + feat + ' < ' + value + ')'
   mid_sqls.append( "{}case when {} then\n".format(indent, cur_sql) )
   parse_xgb_tree_2sql(xgb_tree_json['children'][0], mid_sqls, tree_num, depth+1)
   cur_sql = '(' + feat + ' >= ' + value + ') '
   mid_sqls.append( "\n{}else\n".format(indent) )
   parse_xgb_tree_2sql(xgb_tree_json['children'][1], mid_sqls, tree_num, depth+1)
   mid_sqls.append("\n{}end".format(indent))

elif missing == right_tree:
   cur_sql = '(' + feat + ' is null' + ' or ' + feat + ' >= ' + value + ')'
   mid_sqls.append( "{}case when {} then\n".format(indent, cur_sql) )
   parse_xgb_tree_2sql(xgb_tree_json['children'][1], mid_sqls, tree_num, depth+1)
   cur_sql = '(' + feat + ' < ' + value + ') '
   mid_sqls.append( "\n{}else\n".format(indent) )
   parse_xgb_tree_2sql(xgb_tree_json['children'][0], mid_sqls, tree_num, depth+1)
   mid_sqls.append("\n{}end".format(indent))
else:
   print ("something wrong.")

## 解析模型文件

def parse_xgb_trees(xgb_trees_josn):
   tree_sqls = []
   idx = 0
   for single_tree in xgb_trees_josn:
      mid_sqls = []
      parse_xgb_tree_2sql(json.loads(single_tree), mid_sqls, idx, 0)
      tree_sql = ''
      for t_sql in mid_sqls:
         tree_sql = tree_sql + t_sql
         tree_sql = tree_sql + ' as ' + 'tree_' + str(idx) + '_score,'
         idx += 1
         tree_sqls.append(tree_sql + '\n')
         tree_sqls[-1]=tree_sqls[-1][:-2]
         return tree_sqls

if __name__ == '__main__':

with open("E:\Python\model_XGB_json.json", 'r') as f_read:
   xgb_json = f_read.readlines()
   tree_sqls = parse_xgb_trees(xgb_json)
   final_sqls = ''

for item_sql in tree_sqls:
   final_sqls = final_sqls + item_sql
   with codecs.open("E:\Python\model_XGB_json.sql", 'w', encoding="utf-8") as f:
      for item_sql in tree_sqls:
         f.write(item_sql + '\n')

print (final_sqls)
