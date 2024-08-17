# Bakwaan_Buddy
我们草台班子项目组目前的想法是解决大家计算机学院毕业面临的BUG——不爱背、背不下来八股文，觉得枯燥、烦、工作了用不着，反正就是知识他不进脑子。收藏从未停止，学习从未开始。目前想到设计有2个功能，一个是用户输问题，八股文AI用人话来解释，比如说你问多态的知识，大模型就拿动物类别下面的狗和猫来举例。另一个功能是“随机八股文生成器”，比如说在等地铁啥的，打开AI，点击一个按钮，他就吐一段八股文相关的知识背景和故事。（灵感来源于我学校图书馆的故事机，三个按钮根据需要可以吐出1分钟，3分钟，5分钟的随机故事。那么AI也可以仿照这个设计吐出一段随机八股文来满足利用碎片时间复习的需要。）


使用方法：

1.开一台阿里云实例机器，确保代码文件是保存到一个名为 app.py 的文件中（就只有一个.py后缀的代码文件！！！）。

2.终端输入命令 cd /mnt/workspace/Bakwaan_Buddy/Bakwaan_Buddy 进入app.py所在的外层目录（这个不绝对，看你在电脑上放哪里）

3.在命令行中运行 streamlit run app.py --server.address 127.0.0.1 --server.port 6006。

4.浏览器将自动打开，显示Streamlit应用。
