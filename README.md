# yolopose

## yolo_person.py

### yolo = YOLO_TF(argvs)
#### argvs:
1. '-imshow' 1--True  else--False  Default--True          #detect过后是否展示结果图片
2. '-disp_console' 1--True else--False Default--True      #是否print操作信息

#### main interface
yolo.detect_from_cvmat(self, img):
Parameters:
　　img: a cv mat object
Return:
　　results: a list of results, results[ boxIndex ] [ 0 -- class name; 1 -- x; 2 -- y; 3 -- width; 4 -- height; 5 -- confidence ]

　　 
