close all;
clear;
% author:jdz

% -------------------------- �������� --------------------------
filePath = 'yaleBExtData\';
p = 7; % ����pֵ

number_person = 39; % ���14����ȱʧ����ʵ�����Ϊ38
number_training_per_person = p;% ÿ������ȡ��ѵ����������
height = 192;% ԭʼͼ��߶�
width = 168;% ԭʼͼ����
scale = 4;% ���ű���
faces_train = zeros(1,width/scale*height/scale);
faces_test = zeros(1,width/scale*height/scale);
labels_train = zeros(1,1);
labels_test = zeros(1,1);

% ------------------------- ����Ԥ���� ---------------------------
for i = 1:number_person
  if i==14
    continue;
  end 
  dir_persion_i = sprintf('%s/yaleB%02d/',filePath, i);
  filenames = dir(strcat(dir_persion_i,'yaleB*_P00A*.pgm*'));%������ʵ�����˲���bad�ļ�
  number_pgm = length(filenames);
  fprintf('���%d��%d��ͼƬ\n', i, number_pgm);
  index = randperm(number_pgm);
  
  for j=1:number_pgm
    filename_index = filenames(index(j)).name;  
    %fprintf('filename %s ',filename_index);
    image = imread(strcat(dir_persion_i, filename_index));
    newImage = image(1:scale:end, 1:scale:end);%��������48x42
    [rows, cols] = size(newImage);
    
    if rows==120 %��һ��ͼƬ��ʽ��С��������ͬ,������ѵ���Ͳ���
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

% size(faces_train)��
% %size(faces_train,2)
% size(faces_test)
% size(labels_train)
% size(labels_test)
% ת��Ϊdouble��ʽ
faces_train = double(faces_train(2:end,:)');
faces_test = double(faces_test(2:end,:)');
labels_train = double(labels_train(2:end,:));
labels_test = double(labels_test(2:end,:));


% ------------------------- ģ��ѵ�� -------------------------------
predictLabels=[];
t1=clock;
cout=0;
sigma=0.1;
opts=spgSetParms('verbosity',0);
% ��ÿ�Ų���ͼƬ�������L1����Լ�����̣��õ���Ӧ��ǩ
for i=1:size(faces_test,2)
    fprintf('ѵ����%d��ͼƬ\n',i);
    x(:,i)=spg_bpdn(faces_train,faces_test(:,i),sigma,opts);
    for j=1:number_person
         if j==14
             r(j)=1000000;% �����������14
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
t2=clock;%ѵ��������Ԥ���ǩ������

% -------------------- --ģ�Ͳ��� -------------------------------------
for i=1:length(labels_test)
    if(labels_test(i)==predictLabels(i))
        cout=cout+1;
    end
end
% ��ȷ��
acc=cout/length(labels_test);
t3=clock;
fprintf('��������Ϊ:%d,׼ȷ��Ϊ:%f\n',size(faces_test,2),acc);
fprintf('��ѵ��ʱ��:%f,ƽ��ÿ��ʱ��:%f\n',etime(t2,t1),etime(t2,t1)/size(faces_test,2));
% fprintf('�ܲ���ʱ��:%f,ƽ��ÿ�Ų���ʱ��:%f\n',etime(t3,t2),etime(t3,t2)/size(faces_test,2));
    
