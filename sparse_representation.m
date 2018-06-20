close all;
clear;
% author:jdz

% -------------------------- 变量定义 --------------------------
filePath = 'yaleBExtData\';
p = 7; % 定义p值

number_person = 39; % 类别14数据缺失，故实际类别为38
number_training_per_person = p;% 每个类别抽取的训练样本个数
height = 192;% 原始图像高度
width = 168;% 原始图像宽度
scale = 4;% 缩放比例
faces_train = zeros(1,width/scale*height/scale);
faces_test = zeros(1,width/scale*height/scale);
labels_train = zeros(1,1);
labels_test = zeros(1,1);

% ------------------------- 数据预处理 ---------------------------
for i = 1:number_person
  if i==14
    continue;
  end 
  dir_persion_i = sprintf('%s/yaleB%02d/',filePath, i);
  filenames = dir(strcat(dir_persion_i,'yaleB*_P00A*.pgm*'));%这样其实读入了部分bad文件
  number_pgm = length(filenames);
  fprintf('类别%d有%d张图片\n', i, number_pgm);
  index = randperm(number_pgm);
  
  for j=1:number_pgm
    filename_index = filenames(index(j)).name;  
    %fprintf('filename %s ',filename_index);
    image = imread(strcat(dir_persion_i, filename_index));
    newImage = image(1:scale:end, 1:scale:end);%降采样成48x42
    [rows, cols] = size(newImage);
    
    if rows==120 %第一张图片格式大小与其他不同,不参与训练和测试
        continue
    end
    
    if j <= number_training_per_person
      faces_train = [faces_train; reshape(newImage, 1, rows*cols/1)];
      % fprintf('training %d ',i);
      labels_train = [labels_train; i];
    else
      faces_test = [faces_test; reshape(newImage, 1, rows*cols/1)];
      % fprintf('testing %d ',i);
      labels_test = [labels_test; i];
    end
  end
end

% size(faces_train)；
% %size(faces_train,2)
% size(faces_test)
% size(labels_train)
% size(labels_test)
% 转换为double格式
faces_train = double(faces_train(2:end,:)');
faces_test = double(faces_test(2:end,:)');
labels_train = double(labels_train(2:end,:));
labels_test = double(labels_test(2:end,:));


% ------------------------- 模型训练 -------------------------------
predictLabels=[];
t1=clock;
cout=0;
sigma=0.1;
opts=spgSetParms('verbosity',0);
% 将每张测试图片带入求解L1范数约束方程，得到相应标签
for i=1:size(faces_test,2)
    fprintf('训练第%d张图片\n',i);
    x(:,i)=spg_bpdn(faces_train,faces_test(:,i),sigma,opts);
    for j=1:number_person
         if j==14
             r(j)=1000000;% 不可能是类别14
             continue;
         end
        idx=find(labels_train==j);
        startidx=idx(1,1);
        endidx=idx(end,1);
        % size(x(i,startidx:endidx))
        r(j)=norm(faces_test(:,i)-(faces_train(:,startidx:endidx)*x(startidx:endidx,i)));
    end
    [~,index]=min(r);
    predictLabels=[predictLabels;index];
end
t2=clock;%训练结束，预测标签已生成

% -------------------- --模型测试 -------------------------------------
for i=1:length(labels_test)
    if(labels_test(i)==predictLabels(i))
        cout=cout+1;
    end
end
% 正确率
acc=cout/length(labels_test);
t3=clock;
fprintf('测试张数为:%d,准确率为:%f\n',size(faces_test,2),acc);
fprintf('总训练时间:%f,平均每张时间:%f\n',etime(t2,t1),etime(t2,t1)/size(faces_test,2));
% fprintf('总测试时间:%f,平均每张测试时间:%f\n',etime(t3,t2),etime(t3,t2)/size(faces_test,2));
    
