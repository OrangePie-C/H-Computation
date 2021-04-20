


% Hyper Parameter Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SamplingNumber=4;
Threshold = 55;
Iteration_Number = 150;
Feature_Matching_Threshold=0.7;
    %activation H.P.  {1: True, 2: False}
    Normalization = 2;
    Optimal_estimation = 2;
    Guided_matching = 1;
    matching_Threshold = 150;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
% Initial Image preparation and Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Image Loading
H1_ex1 = imread("H1_ex1.png");
H1_ex2 = imread("H1_ex2.png");
H2_ex1 = imread("H2_ex1.png");
H2_ex2 = imread("H2_ex2.png");

%Pixel value save
H1_ex1_Gray = rgb2gray(H1_ex1);
H1_ex2_Gray = rgb2gray(H1_ex2);
H2_ex1_Gray = rgb2gray(H2_ex1);
H2_ex2_Gray = rgb2gray(H2_ex2);

%corner points
corners_H1_1 = detectHarrisFeatures(H1_ex1_Gray);
corners_H1_2 = detectHarrisFeatures(H1_ex2_Gray);
corners_H2_1 = detectHarrisFeatures(H2_ex1_Gray);
corners_H2_2 = detectHarrisFeatures(H2_ex2_Gray);
%corners_H2_1 = detectSURFFeatures(H2_ex1_Gray);
%corners_H2_2 = detectSURFFeatures(H2_ex2_Gray);

[features1, valid_corners1] = extractFeatures(H1_ex1_Gray, corners_H1_1);
[features2, valid_corners2] = extractFeatures(H1_ex2_Gray, corners_H1_2);
%indexPairs = matchFeatures(features1,features2);
indexPairs = FeatureMatching(features1.Features,features2.Features, Feature_Matching_Threshold);
%indexPairs = FeatureMatching(features1,features2, Feature_Matching_Threshold);
matchedPoints1 = valid_corners1(indexPairs(:,1));
matchedPoints2 = valid_corners2(indexPairs(:,2));
matchedPoints_int = [double(indexPairs(:,1)),matchedPoints1.Location,matchedPoints2.Location,zeros(size(matchedPoints1,1),1)];
matchedPoints_int2 = matchedPoints_int;
samplePoints = zeros(SamplingNumber,5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











% Sampling and Iteration beginning point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

best_count = 0;
for iteration = 1:Iteration_Number

    count = 0;
    while count < SamplingNumber
        NumberOfMatches = size(matchedPoints1.Location);
        temp= ceil(rand * NumberOfMatches(1));

        newArray = [temp,matchedPoints1.Location(temp,1),matchedPoints1.Location(temp,2),matchedPoints2.Location(temp,1),matchedPoints2.Location(temp,2)];

        if newArray(1) ~= samplePoints(:,1)
            count = count +1;
            samplePoints(count,:) = newArray;
        end
        count;
    end
    
    
    %A calculation
    A = zeros(SamplingNumber*2,9);
    for i = 1:SamplingNumber
        subA = zeros(2,9);

        subA(1,1) = samplePoints(i,2);
        subA(1,2) = samplePoints(i,3);
        subA(1,3) = 1;
        subA(1,7) = -1*samplePoints(i,4)*samplePoints(i,2);
        subA(1,8) = -1*samplePoints(i,4)*samplePoints(i,3);
        subA(1,9) = -1*samplePoints(i,4);

        subA(2,4) = samplePoints(i,2);
        subA(2,5) = samplePoints(i,3);
        subA(2,6) = 1;
        subA(2,7) = -1*samplePoints(i,5)*samplePoints(i,2);
        subA(2,8) = -1*samplePoints(i,5)*samplePoints(i,3);
        subA(2,9) = -1*samplePoints(i,5);

        A(2*i-1,:) = subA(1,:);
        A(2*i,:) = subA(2,:);
    end
    [U,S,V] = svd(A);
    H=V(:,9);
    H=reshape(H,[3,3])';
    qH=H;
    
    %T1,T2 calculation
    mean_num= mean(samplePoints);
    numer1 = 0;
    numer2 = 0;
    for i = 1:4
        numer1 = numer1 + sqrt(((samplePoints(i,2)-mean_num(2))^2)+((samplePoints(i,3)-mean_num(3))^2));
        numer2 = numer2 + sqrt(((samplePoints(i,4)-mean_num(4))^2)+((samplePoints(i,5)-mean_num(5))^2));
    end
    s1=(sqrt(2)*SamplingNumber)/numer1;
    s2=(sqrt(2)*SamplingNumber)/numer2;
    t1_x = -1*s1*mean_num(2);
    t1_y = -1*s1*mean_num(3);
    t2_x = -1*s2*mean_num(4);
    t2_y = -1*s2*mean_num(5);
    T1=[s1,0,t1_x;0,s1,t1_y;0,0,1];
    T2=[s2,0,t2_x;0,s2,t2_y;0,0,1];
    %A calc w/ normalized
    n_samplePoints=zeros(SamplingNumber,5);
    n_samplePoints(:,1) = samplePoints(:,1);
    for i = 1:SamplingNumber
        n_samplePoints(i,2) = (s1*samplePoints(i,2))+t1_x;
        n_samplePoints(i,3) = (s1*samplePoints(i,3))+t1_y;
        n_samplePoints(i,4) = (s2*samplePoints(i,4))+t2_x;
        n_samplePoints(i,5) = (s2*samplePoints(i,5))+t2_y;
    end
    An = zeros(SamplingNumber*2,9);
    for i = 1:SamplingNumber
        subA = zeros(2,9);

        subA(1,1) = n_samplePoints(i,2);
        subA(1,2) = n_samplePoints(i,3);
        subA(1,3) = 1;
        subA(1,7) = -1*n_samplePoints(i,4)*n_samplePoints(i,2);
        subA(1,8) = -1*n_samplePoints(i,4)*n_samplePoints(i,3);
        subA(1,9) = -1*n_samplePoints(i,4);

        subA(2,4) = n_samplePoints(i,2);
        subA(2,5) = n_samplePoints(i,3);
        subA(2,6) = 1;
        subA(2,7) = -1*n_samplePoints(i,5)*n_samplePoints(i,2);
        subA(2,8) = -1*n_samplePoints(i,5)*n_samplePoints(i,3);
        subA(2,9) = -1*n_samplePoints(i,5);

        An(2*i-1,:) = subA(1,:);
        An(2*i,:) = subA(2,:);
    end
    [U2,S2,V2] = svd(An);
    H_n_temp=V2(:,9);
    H_n_temp=reshape(H_n_temp,[3,3])';
    H_n = inv(T2)*H_n_temp*T1;
    if Normalization ==1
        final_H = H_n;
    else
        final_H = H;
    end
    %calculate distance
    count = 0;
    for i = 1:size(matchedPoints_int,1)
        x=matchedPoints_int(i,2);
        y=matchedPoints_int(i,3);
        new_coor = final_H * [x;y;1];
        new_coor = new_coor/new_coor(3);
        matchedPoints_int(i,6)=sqrt(((new_coor(1)-matchedPoints_int(i,4))^2)+((new_coor(2)-matchedPoints_int(i,5))^2));
        if matchedPoints_int(i,6)<Threshold
            count = count +1;
        end
    end
    if count>best_count
        best_count = count;
        bestH = final_H;
        best_matchedPoints_int = matchedPoints_int;
        selected_samples = samplePoints;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Additional Function: Optimal_Estimation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global tempiii;
tempiii = best_matchedPoints_int;

if Optimal_estimation == 1
    Original_H = bestH;
    options.Algorithm = 'levenberg-marquardt';
    options.InitDamping = 0.1;
    options.StepTolerance = 1.000000e-09;
    
    for i = size(tempiii,1):-1:1
        x=tempiii(i,2);
        y=tempiii(i,3);
        new_x= ((bestH(1,1)*x)+(bestH(1,2)*y)+(bestH(1,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
        new_y= ((bestH(2,1)*x)+(bestH(2,2)*y)+(bestH(2,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
        k = sqrt(((tempiii(i,4)-new_x)^2)+((tempiii(i,5)-new_y)^2));
        tempii(i,6)=k;
        if k>=35
            tempiii(i,:)=[];
        end
    end

    [Refined_H,resnorm,residual,exitflag,output] = lsqnonlin(@extraction,bestH,[],[],options);

    bestH=xk;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Additional Function: Guided_matching %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Guided_matching ==1
    for i = 1: size(best_matchedPoints_int,1)
        if best_matchedPoints_int(i,6) >= Threshold
            temp_feature2_index=zeros(0,1);
            temp_feature2=zeros(0,64);
            %x=best_matchedPoints_int(i,2);
            %y=best_matchedPoints_int(i,3);
            tempi=bestH*[best_matchedPoints_int(i,2);best_matchedPoints_int(i,3);1];
            best_matchedPoints_int(i,2);
            best_matchedPoints_int(i,3);
            new_x = tempi(1)/tempi(3);
            new_y = tempi(2)/tempi(3);
            %new_x= ((bestH(1,1)*x)+(bestH(1,2)*y)+(bestH(1,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
            %new_y= ((bestH(2,1)*x)+(bestH(2,2)*y)+(bestH(2,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
            
            for j = 1: size(valid_corners2.Location,1)
                if sqrt(((new_x-valid_corners2.Location(j,1))^2)+(new_y-valid_corners2.Location(j,2))^2)<matching_Threshold
                   temp_feature2_index = [temp_feature2_index;j ];
                   temp_feature2 = [temp_feature2; features2.Features(j,:) ];
                   %temp_feature2 = [temp_feature2; features2(j,:) ];
                end 
            end
            %여기에 문제가 있는게 아닌가...
            indexPairs2 = [];
            if size(temp_feature2,1) ~= 0
            indexPairs2 = FeatureMatching(features1.Features(i,:),temp_feature2, 1);
            %indexPairs2 = FeatureMatching(features1(i,:),temp_feature2, 1);
            end
            if size(indexPairs2,1) ~= 0 
                
                best_matchedPoints_int(i,4)= valid_corners2.Location(temp_feature2_index(indexPairs2(1,2)),1);
                best_matchedPoints_int(i,5)= valid_corners2.Location(temp_feature2_index(indexPairs2(1,2)),2);
                best_matchedPoints_int(i,6)= sqrt(((new_x-best_matchedPoints_int(i,4))^2)+(new_y-best_matchedPoints_int(i,5))^2)  ;
            end
        end
     end
end

best_count_final = 0;
for i = 1:size(best_matchedPoints_int,1)
    if best_matchedPoints_int(i,6)<Threshold
        best_count_final= best_count_final+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Functions Pool %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tindexPairs=FeatureMatching(a,b,c)

    tindexPairs=zeros(0,2);
    for i = 1: size(a,1)

        if rem(i,100) ==0
            disp(round(double(i/size(a,1))*100))
        end
        temp_storage=zeros(0,2);
        for j = 1: size(b,1)
            temp_storage(j,1) = j;
            temp_storage(j,2) = sum(abs(double(b(j,:))-double(a(i,:))));
        end
        temp_storage=sortrows(temp_storage,2);
        if (temp_storage(1,2)/temp_storage(2,2))<c
            tindexPairs=[tindexPairs;[i,temp_storage(1,1)]];

        end

    end
end


function a = extraction(bestH)
    global tempiii
    
        x=tempiii(:,2);
        y=tempiii(:,3);
        
        new_x= ((bestH(1,1)*x)+(bestH(1,2)*y)+(bestH(1,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
        new_y= ((bestH(2,1)*x)+(bestH(2,2)*y)+(bestH(2,3)))/((bestH(3,1)*x)+(bestH(3,2)*y)+(bestH(3,3)));
            
        a=double(sqrt(((tempiii(:,4)-new_x).^2)+((tempiii(:,5)-new_y).^2)));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

