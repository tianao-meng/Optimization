classdef Neural_Network < handle
    properties 
        inputsize = 2;
        outputsize = 1;
        hiddensize = 6;
        
        W1 = normrnd(0,1,2,5);
        W2 = normrnd(0,1,5,9);
        W3 = normrnd(0,1,9,5);
        W4 = normrnd(0,1,5,3);
        W5 = normrnd(0,1,3,1);
        
        W = [0 ; 0]
        der_total = []; %total derivative norm
        
        b1 = normrnd(0,1,1,5);
        b2 = normrnd(0,1,1,9);
        b3 = normrnd(0,1,1,5);
        b4 = normrnd(0,1,1,3);
        b5 = normrnd(0,1,1,1);
        
        b_con = 0;
        r;
        Ap;
        

    end
    methods
        %Forard to get prediction
        function obj = Neural_Network()
        end

        
        function output = Forward(obj, x)
            %input -> first hiden lier

            z = x * obj.W1 + obj.b1;
            size_z = size(z);
            z2 = [];
            
            for i = 1 : size_z(2)
                
                res = obj.relu_activate(z(i));
                z2 = [z2 res];
            end
            
            
            %first hiden lier -> second hiden lier
            z3 = z2 * obj.W2 + obj.b2;
            size_z3 = size(z3);
            z4 = [];
            for i = 1 : size_z3(2)
                res = obj.relu_activate(z3(i));
                z4 = [z4 res];
            end
            
            
            %second hiden lier -> third hiden lier
            z5 = z4 * obj.W3 + obj.b3;
            size_z5 = size(z5);
            z6 = [];
            for i = 1 : size_z5(2)
                res = obj.relu_activate(z5(i));
                z6 = [z6 res];
            end
            
            
            %third hiden lier -> fourth hiden lier
            z7 = z6 * obj.W4 + obj.b4;
            size_z7 = size(z7);
            z8 = [];
            for i = 1 : size_z7(2)
                res = obj.relu_activate(z7(i));
                z8 = [z8 res];
            end
            
            
            %fourth hiden lier -> output
            z9 = z8 * obj.W5 + obj.b5;
            
            size_z9 = size(z9);
            predict = [];
            for i = 1 : size_z9(2)
                res = obj.relu_activate(z9(i));
                predict = [predict res];
            end    
            

            output = {z, z2, z3, z4, z5, z6, z7, z8, z9, predict};
            

        end
        
        function der_theta = derivative_ann(obj, x, y, output)
            error = output{10} - y;
            
            output_error_delta = 2 * error; % error / predict
            %output{9}
            
            z9_delata = output_error_delta * obj.activate_derivative(output{9}); %error / z9
            
  
            z8_error = z9_delata * transpose(obj.W5);%error / z8
            
           
            
            z8_b_error = z9_delata; 
            z7_delata = z8_error .* obj.activate_derivative(output{7});%error / z7
            
            
            
            z6_error = z7_delata * transpose(obj.W4); %error / z6
            z6_b_error = z7_delata;
            z5_delata = z6_error .* obj.activate_derivative(output{5});%error / z5
            
           
            %obj.activate_derivative(output{5})
            
            z4_error = z5_delata * transpose(obj.W3);%error / z4
            z4_b_error = z5_delata;
            z3_delata = z4_error .* obj.activate_derivative(output{3});%error / z3
            
            z2_error = z3_delata * transpose(obj.W2);%error / z2
            z2_b_error = z3_delata;
            z1_delata = z2_error .* obj.activate_derivative(output{1});%error / z
            
            %x_error = dot(z1_delata, transpose(obj.W1));
            x_b_error = z1_delata;
            
            
            der_W1 = transpose(x) * z1_delata ;
            der_b1 = x_b_error;
            der_W2 = transpose(output{2}) * z3_delata ;
            der_b2 = z2_b_error;
            der_W3 = transpose(output{4}) * z5_delata ;
            der_b3 = z4_b_error;
            der_W4 = transpose(output{6}) * z7_delata ;
            der_b4 = z6_b_error;
            der_W5 = transpose(output{8}) * z9_delata ;
            der_b5 = z8_b_error;
            %der_b1 = x_b_error;
            der = [norm(der_W1) norm(der_b1) norm(der_W2) norm(der_b2) norm(der_W3) norm(der_b3) norm(der_W4) norm(der_b4) norm(der_W5) norm(der_b5)];
            der_theta = {der_W5, z8_b_error, der_W4, z6_b_error, der_W3, z4_b_error, der_W2, z2_b_error, der_W1, x_b_error};
            obj.der_total = norm(der);
        end
            
        
        function backward_gradient_decent(obj, x, y, output, learning_rate)
            der_theta = derivative_ann(obj, x, y, output);

            obj.W1 = obj.W1 - learning_rate * der_theta{9};
            obj.b1 = obj.b1 - learning_rate * der_theta{10};

            obj.W2 = obj.W2 - learning_rate * der_theta{7};
            obj.b2 = obj.b2 - learning_rate * der_theta{8};
            

            obj.W3 = obj.W3 - learning_rate * der_theta{5};
            obj.b3 = obj.b3 - learning_rate * der_theta{6};
            
            
  
            obj.W4 = obj.W4 - learning_rate * der_theta{3};
            obj.b4 = obj.b4 - learning_rate * der_theta{4};
            

            obj.W5 = obj.W5 - learning_rate * der_theta{1};
            obj.b5 = obj.b5 - learning_rate * der_theta{2};
            

        end
        
        function conjgrad(obj, A, observation, learning_rate)
            %x = obj.W;
            b = 2 * (obj.b_con -observation) * obj.W;
            r = 2 .* b - 2 .* A * obj.W;
            p = r;
            rsold = -2 .* A .* r;

            for i = 1:length(b)
                Ap = A .* p;
                %alpha = rsold / (-2 * A * Ap);
                obj.W = obj.W + learning_rate .* p;
                r = r - learning_rate .* Ap;
                rsnew = -2 .* A .* r;
                if rsnew .^ 0.5 < 0.00000000001
                      break;
                end
                p = r + (rsnew / rsold) * p;
                rsold = rsnew;
            end
        end
        
        function conjgrad_backtracking(obj, A, observation, learning_rate)
            
            %x = obj.W;
            b = 2 * (obj.b_con -observation) * obj.W;
            obj.r = 2 .* b - 2 .* A * obj.W;
            p = obj.r;
            rsold = -2 .* A .* obj.r;

            for i = 1:length(b)
                obj.Ap = A .* p;
                %alpha = rsold / (-2 * A * Ap);
                obj.W = obj.W + learning_rate .* p;
                obj.r = obj.r - learning_rate .* obj.Ap;
                rsnew = -2 .* A .* obj.r;
                if rsnew .^ 0.5 < 0.00000000001
                      break;
                end
                p = obj.r + (rsnew / rsold) * p;
                rsold = rsnew;
            end
        end
        
        
        
        
            
         %relu activation function   
        function x_new = relu_activate(obj,input)
            
            
            if input <= 0
                x_new = 0;
            else
                x_new = input;
            end
            
           

        end
        %relu activation function derivative
        function act_der = activate_derivative(obj, input)
            size_input = size(input);
            act_der = [];
            for i = 1 : size_input(2)
                if input(i) <= 0
                    res = 0;
                    act_der = [act_der res];
                else
                    res = 1;
                    act_der = [act_der res];
                end
            end
        end
        
        
        function train_gradient_decent_fix_step(obj, x, y)
            learning_rate = 0.0001;
            output = obj.Forward(x);
            obj.backward_gradient_decent(x, y, output, learning_rate);
            
            %output
            
        end
        
        function output = train_gradient_decent_backtracking(obj, x, y)
            learning_rate = 1;
            output_original = obj.Forward(x);
            der_theta = obj.derivative_ann(x, y, output_original);
            obj.backward_gradient_decent(x, y, output_original, learning_rate);

            output_new = obj.Forward(x);
            
            while  (output_new{10} - y) * (output_new{10} - y) > ((output_original{10} - y) * (output_original{10} - y) + (learning_rate / 2) * (obj.der_total) * (obj.der_total))
                
                learning_rate
                if learning_rate == 0
                    break;
                end
                
                obj.W1 = obj.W1 + learning_rate * der_theta{9};
                obj.W2 = obj.W2 + learning_rate * der_theta{7};
                obj.W3 = obj.W3 + learning_rate * der_theta{5};
                obj.W4 = obj.W4 + learning_rate * der_theta{3};
                obj.W5 = obj.W5 +  learning_rate * der_theta{1};
                
                obj.b1 = obj.b1 + learning_rate * der_theta{10};
                obj.b2 = obj.b2 + learning_rate * der_theta{8};
                obj.b3 = obj.b3 + learning_rate * der_theta{6};
                obj.b4 = obj.b4 + learning_rate * der_theta{4};
                obj.b5 = obj.b5 + learning_rate * der_theta{2};
                learning_rate = 0.5 * learning_rate;
                %learning_rate
                obj.W1 = obj.W1 - learning_rate * der_theta{9};
                obj.W2 = obj.W2 - learning_rate * der_theta{7};
                obj.W3 = obj.W3 - learning_rate * der_theta{5};
                obj.W4 = obj.W4 - learning_rate * der_theta{3};
                obj.W5 = obj.W5 - learning_rate * der_theta{1};
                
                obj.b1 = obj.b1 - learning_rate * der_theta{10};
                obj.b2 = obj.b2 - learning_rate * der_theta{8};
                obj.b3 = obj.b3 - learning_rate * der_theta{6};
                obj.b4 = obj.b4 - learning_rate * der_theta{4};
                obj.b5 = obj.b5 - learning_rate * der_theta{2};
                output_new = obj.Forward(x);
                %output_new
                

            end
            
            output = output_new;
            
            
        end

        % (A * theta + b - y)^2
        %theta * A * At * theta + 2 * (b - y) * A * theta + (b - y) ^ 2
        %2 * A * theta + 2 * (b - y) * A 
        % A = 2 * x
        % b = 2 * (b - y) * x
        function train_conjugate_gradient_fix_step(obj, x, observation)
            learning_rate = 0.00001;
            
            %obj.W = obj.W1 * obj.W2 * obj.W3 * obj.W4 * obj.W5;
            
            A = 2 * x;
            %b = obj.b_con;
            
            
            obj.conjgrad(A,observation, learning_rate);
            
            
        end
        
        function output = train_conjugate_gradient_backtracking(obj, x, y)
            learning_rate = 1;
            output_original = obj.W * x + obj.b_con;
            A = 2 * x;
            obj.conjgrad(A,y, learning_rate);
            output_new = obj.W * x + obj.b_con;
            
            while  (output_new - y) * (output_new - y) > ((output_original - y) * (output_original - y) + (learning_rate / 2) * 4 * x * transpose(x))
                
                learning_rate
                if learning_rate == 0
                    break;
                end
                
                obj.W = obj.W - learning_rate .* p;
                obj.r = obj.r + learning_rate .* obj.Ap;
                learning_rate = 0.5 * learning_rate;
                obj.conjgrad(A,y, learning_rate);
                output_new = obj.W * x + obj.b_con;
                %output_new
                

            end
            
            output = output_new;
            
            
        end
        
    
    
    end
end
