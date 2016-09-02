% Returns our neural network connectivity structure.
% --Ayan Chakrabarti <ayanc@ttic.edu>
function nw = nw_struct

nw = { 
% Level 1 

    {'FCR', {'inp1','inp2'}, 'l1_1', []};  
    {'FCR', {'inp2','inp3'}, 'l1_2', []};  
    {'FCR', {'inp3','inp4'}, 'l1_3', []};  

% Level 2

    {'FCR', {'l1_1','l1_2'}, 'l2_1', []};  
    {'FCR', {'l1_2','l1_3'}, 'l2_2', []};  

% Globals

    {'FCR', {'l2_1','l2_2'}, 'gb_1', []};  
    {'FCR', {'gb_1'}, 'gb_2', []};  
    {'FCR', {'gb_2'}, 'gb_3', []};  
    {'FCR', {'gb_3'}, 'gb_4', []};  
    {'FCR', {'gb_4'}, 'gb_5', []};  

% Final output layer
    {'FC', {'gb_5'}, 'output', []};
};


nw = makeNWDef(nw,{'inp1','inp2','inp3','inp4'});