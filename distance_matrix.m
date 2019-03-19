function D = distance_matrix(M_data, M_ref)

N = size(M_data,2);  MM = size(M_ref,2);
D = ((sum(M_ref'.^2,2)*ones(1,N))+(sum(M_data'.^2,2)*ones(1,MM))'-(2*(M_ref'*M_data)));
    