/*
*
* Madhav Datt
* Program to prepare random numberlink/flow puzzle gameboards using Disjoint set Data Structures
*
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

//data type representing a node
typedef struct node
{
	struct node* parent;
	int rank, path_num;
	bool endpoint;
} node;

//prototyping for function to print board - unsolved
void printboard (node*** B, int n, bool solved);

//function to initialize board with grid size n X n
node*** initboard (int n)
{
	node*** B = new node**[n];
	int i, j;
	for (i = 0; i < n; i++)
		B[i] = new node*[n];

	// cout << "\n+++ Initializing board...\n\n";
		
	//allocating space and initializing variables
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			B[i][j] = new node;
			B[i][j] -> rank = 0;
			B[i][j] -> path_num = 0;
			B[i][j] -> parent = NULL;
			B[i][j] -> endpoint = 0;
		}
	return B;
}

//function to find set of a given node
node* findset (node* x)
{
	if (x -> parent != NULL)
		x = findset (x -> parent);
	return x;
}

//function to merge two disjoint sets of nodes
void setunion (node* x, node* y)
{
	node* set_x = findset (x);
	node* set_y = findset (y);
	
	//merging smaller tree into larger
	if (set_x -> rank > set_y -> rank)
		set_y -> parent = set_x;
	else 
		set_x -> parent = set_y;
	
	//increasing rank if both ranks were equal
	if (set_x -> rank == set_y -> rank)
		set_y -> rank += 1;
}

//function to check if B(k, l) has more than 1 neighbour that is part of current incomplete path
bool check_nbr (node*** B, int i, int j, int k, int l, int n)
{
	int a, b, count;
	count = 0;	
	
	//north neighbour
	a = k - 1;
	b = l;
	if(a >= 0 && findset (B[a][b]) == findset (B[i][j])) 
		count++;	
	
	//east neighbour
	a = k;
	b = l + 1;
	if(b < n && findset (B[a][b]) == findset(B[i][j])) 
		count++;	
	
	//west neighbour
	a = k;
	b = l - 1;
	if(b >= 0 && findset (B[a][b]) == findset (B[i][j])) 
		count++;	
	
	//south neighbour
	a = k + 1;
	b = l;
	if(a < n && findset (B[a][b]) == findset (B[i][j])) 
		count++;
	
	if (count > 1) 
		return 0;
	else 
		return 1;
}

//function to add one path to board
bool addpath (node*** B, int n, int path_count, int seed)
{
	int i, j, i1, j1, n_order, k, l, status, count;
	
	//adding a path with random empty starting coordinates
	i = rand () % n;
	j = rand () % n;
	k = -1;
	l = -1;

	status = 0;
	count = 0;
	
	//finding block just before (i, j) in row-major wrap
	if (i == 0 && j == 0)
	{
		i1 = n - 1;
		j1 = n - 1;
	}
	else if (j == 0)
	{
		i1 = i - 1;
		j1 = n - 1;
	}
	else
	{
		i1 = i;
		j1 = j - 1;	
	}
	
	//finding eligible empty block to start from
	while (true)
	{
		
		//if empty block found
		if (findset (B[i][j]) -> rank == 0)
		{
			//checking if empty neighbour exists
			n_order = rand () % 3;
			
			//checking in NEWS order
			if (n_order == 0)
			{
				//north neighbour
				if (i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0)
					{
						k = i + 1;
						l = j;
						break;
					}
				
				//east neighbour
				if (j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0)
					{
						k = i;
						l = j + 1;
						break;
					}
				
				//west neighbour
				if (j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0)
					{
						k = i;
						l = j - 1;
						break;
					}
				
				//south neighbour
				if (i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0)
					{
						k = i - 1;
						l = j;
						break;
					}
			}
			
			//checking in EWSN order
			else if (n_order == 1)
			{		
				//east neighbour
				if (j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0)
					{
						k = i;
						l = j + 1;
						break;
					}
				
				//west neighbour
				if (j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0)
					{
						k = i;
						l = j - 1;
						break;
					}
				
				//south neighbour
				if (i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0)
					{
						k = i - 1;
						l = j;
						break;
					}
				
				//north neighbour
				if (i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0)
					{
						k = i + 1;
						l = j;
						break;
					}
			}
			
			//checking in WSNE order
			else if (n_order == 2)
			{	
				//west neighbour
				if (j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0)
					{
						k = i;
						l = j - 1;
						break;
					}
				
				//south neighbour
				if (i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0)
					{
						k = i - 1;
						l = j;
						break;
					}
				
				//north neighbour
				if (i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0)
					{
						k = i + 1;
						l = j;
						break;
					}
				
					
				//east neighbour
				if (j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0)
					{
						k = i;
						l = j + 1;
						break;
					}
			}
			
			//checking in SNEW order
			else if (n_order == 3)
			{					
				//south neighbour
				if (i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0)
					{
						k = i - 1;
						l = j;
						break;
					}
				
				//north neighbour
				if (i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0)
					{
						k = i + 1;
						l = j;
						break;
					}
				
				//east neighbour
				if (j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0)
					{
						k = i;
						l = j + 1;
						break;
					}
				
				//west neighbour
				if (j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0)
					{
						k = i;
						l = j - 1;
						break;
					}
				
			}
			
		}	
		
		//if wrapped around to where it started
		if (count >= n * n)
			return 0;
		
		//iterating ahead to check next block
		j++;
		count++;
		
		//wrapping around rows
		if (j == n)
		{
			i++;
			if (i == n)
				i = 0;
			j = 0;
		}
	}

	B[i][j] -> endpoint = 1;
	setunion (B[i][j], B[k][l]);
	if(path_count == 0){
		printf ("\"%d\": [[%d, %d], [%d, %d]", path_count, i, j, k, l);
	}else{
		printf (", \"%d\": [[%d, %d], [%d, %d]", path_count, i, j, k, l);
	}
	
	//repeating till path can be extended
	while (true)
	{
		// printf(", ");
		i = k;
		j = l;
		status = 0;
		
			//checking if empty neighbour exists
			//checking if empty neighbour isn't next to another block on same incomplete path
			n_order = rand () % 3;
			
			//checking in NEWS order
			if (n_order == 0)
			{
				//north neighbour
				if (!status && i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0 && check_nbr (B, i, j, i + 1, j, n))
					{
						k = i + 1;
						l = j;
						status = 1;
					}
				
				//east neighbour
				if (!status && j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0 && check_nbr (B, i, j, i, j + 1, n))
					{
						k = i;
						l = j + 1;
						status = 1;
					}
				
				//west neighbour
				if (!status && j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0 && check_nbr (B, i, j, i, j - 1, n))
					{
						k = i;
						l = j - 1;
						status = 1;
					}
				
				//south neighbour
				if (!status && i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0 && check_nbr (B, i, j, i - 1, j, n))
					{
						k = i - 1;
						l = j;
						status = 1;
					}
			}
			
			//checking in EWSN order
			else if (n_order == 1)
			{		
				//east neighbour
				if (!status && j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0 && check_nbr (B, i, j, i, j + 1, n))
					{
						k = i;
						l = j + 1;
						status = 1;
					}
				
				//west neighbour
				if (!status && j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0 && check_nbr (B, i, j, i, j - 1, n))
					{
						k = i;
						l = j - 1;
						status = 1;
					}
				
				//south neighbour
				if (!status && i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0 && check_nbr (B, i, j, i - 1, j, n))
					{
						k = i - 1;
						l = j;
						status = 1;
					}
				
				//north neighbour
				if (!status && i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0 && check_nbr (B, i, j, i + 1, j, n))
					{
						k = i + 1;
						l = j;
						status = 1;
					}
			}
			
			//checking in WSNE order
			else if (n_order == 2)
			{	
				//west neighbour
				if (!status && j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0 && check_nbr (B, i, j, i, j - 1, n))
					{
						k = i;
						l = j - 1;
						status = 1;
					}
				
				//south neighbour
				if (!status && i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0 && check_nbr (B, i, j, i - 1, j, n))
					{
						k = i - 1;
						l = j;
						status = 1;
					}
				
				//north neighbour
				if (!status && i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0 && check_nbr (B, i, j, i + 1, j, n))
					{
						k = i + 1;
						l = j;
						status = 1;
					}
				
				//east neighbour
				if (!status && j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0 && check_nbr (B, i, j, i, j + 1, n))
					{
						k = i;
						l = j + 1;
						status = 1;
					}
			}
			
			//checking in SNEW order
			else if (n_order == 3)
			{					
				//south neighbour
				if (!status && i - 1 >= 0)
					if (findset (B[i - 1][j]) -> rank == 0 && check_nbr (B, i, j, i - 1, j, n))
					{
						k = i - 1;
						l = j;
						status = 1;
					}
				
				//north neighbour
				if (!status && i + 1 < n)
					if (findset (B[i + 1][j]) -> rank == 0 && check_nbr (B, i, j, i + 1, j, n))
					{
						k = i + 1;
						l = j;
						status = 1;
					}
					
				//east neighbour
				if (!status && j + 1 < n)
					if (findset (B[i][j + 1]) -> rank == 0 && check_nbr (B, i, j, i, j + 1, n))
					{
						k = i;
						l = j + 1;
						status = 1;
					}
				
				//west neighbour
				if (!status && j - 1 >= 0)
					if (findset (B[i][j - 1]) -> rank == 0 && check_nbr (B, i, j, i, j - 1, n))
					{
						k = i;
						l = j - 1;
						status = 1;
					}
			}
		
		//if no such neighbour found - break loop
		if (!status)
			break;		

		setunion (B[i][j], B[k][l]);
		printf (", [%d, %d]", k, l);
	}
	
	//sending endpoint
	B[k][l] -> endpoint = 1;
	cout << "]";
	return 1;
}

//function to assign path values (numbers) to filled blocks
void addpathnum (node*** B, int n)
{
	int i, j, pathnum;
	pathnum = 1;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
		{
			//checking if block is non-empty
			if (findset (B[i][j]) -> rank != 0)
			{
				//if root of path tree has not been assigned pathnum
				if (findset (B[i][j]) -> path_num == 0)
				{
					findset (B[i][j]) -> path_num = pathnum;
					B[i][j] -> path_num = pathnum;
					pathnum++;
				}
				
				//if root already has an assigned pathnum
				else
				{
					B[i][j] -> path_num = findset (B[i][j]) -> path_num;
				}
			}
		}
}

//function to print generated puzzle and solution - X for solid, . for blank, numbers for end points and in solution
//function prints formatted output where formatting works best for 1 digit path numbers
void printboard (node*** B, int n, bool solved)
{
	int i, j;
	if (!solved)
	{
		for (i = 0; i < n; i++)
		{
			//for fomatting purposes
			cout << "\n+";
			for (j = 0; j < n; j++)
				cout << "---+";
			cout << "\n|";
			
			for (j = 0; j < n; j++)
			{
				if (B[i][j] -> path_num == 0)
					cout << " X |";
				else if (B[i][j] -> endpoint != 1)
					cout << " . |";
				else
					cout << " " << B[i][j] -> path_num << " |";
			}
		}
		
		//for formatting purposes
		cout << "\n+";
		for (j = 0; j < n; j++)
			cout << "---+";
		cout << "\n";
	}
	
	else
	{
		for (i = 0; i < n; i++)
		{
			//for fomatting purposes
			cout << "\n+";
			for (j = 0; j < n; j++)
				cout << "---+";
			cout << "\n|";
			
			for (j = 0; j < n; j++)
			{
				if (B[i][j] -> path_num == 0)
					cout << " X |";
				else
					cout << " " << B[i][j] -> path_num << " |";
			}
		}
		
		//for fomatting purposes
		cout << "\n+";
		for (j = 0; j < n; j++)
			cout << "---+";
		cout << "\n";
	}
	cout <<"\n";
}

int main (int argc, char* argv[])
{
	if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "--destination DESTINATION SOURCE" << std::endl;
        return 1;
    }
    std::vector <std::string> sources;
    std::string boardSize, level, nExamples, sSeed;
	int ok;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-b") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                boardSize = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "-b option requires one argument." << std::endl;
                return 1;
            }  
        } else if (std::string(argv[i]) == "-lvl") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                level = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "-lvl option requires one argument." << std::endl;
                return 1;
            }  
        } else if (std::string(argv[i]) == "-n") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                nExamples = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "-n option requires one argument." << std::endl;
                return 1;
            }  
        }else if (std::string(argv[i]) == "-s") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                sSeed = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  std::cerr << "-s option requires one argument." << std::endl;
                return 1;
            }  
        }else {
            sources.push_back(argv[i]);
        }
    }

	
	int iBoardSize, iLvl, iN, iSeed;
	// cout << "Enter n\n";
	// cin >> n;

	iBoardSize = stoi(boardSize);
	iLvl = stoi(level);
	iN = stoi(nExamples);
	iSeed = stoi(sSeed);

	srand ((unsigned int) iSeed );
	// path_count = 0;

	if (iN < 1)
	{
		cout << "Invalid value of n. n must be greater than 0\n\n";
		return 0;
	}

	if (iSeed < 1)	
	{
		cout << "Invalid value of seed (s). s must be greater than 0\n\n";
		return 0;
	}
	//initializing board
	node*** B = initboard (iBoardSize);
	
	//adding paths till possible
	// cout << "+++ Adding paths...\n\n";
	printf ("{");
	for(int path_count = 0; path_count < iLvl && addpath (B, iBoardSize, path_count, iSeed); path_count++){
		// addpath (B, n, path_count++)
	}
	// while (addpath (B, n, path_count++));
	printf ("}");
	//assigning path numbers
	// cout << "\n+++ Assigning path numbers...\n\n";
	// addpathnum (B, n);


	
	//printing puzzle and solution
	//printing with solved parameter as 0 to get puzzle only
	// cout << "+++ The puzzle...\n\n";
	// printboard (B, n, 0);
	
	// //printing with solved parameter as 1 to get solution
	// cout << "+++ The solution...\n\n";
	// printboard (B, n, 1);
	
	return 0;
}
