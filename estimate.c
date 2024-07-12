#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int n, k, m;
double **trainX;
double **trainY;
double **inputX;

void printMatrix(double **ar, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%lf  ", ar[i][j]);
        }
        printf("\n");
    }
}

int readTrainingFile(char *filename)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
        return 1;

    char line[256];
    fscanf(file, "%s", line);
    if (!strcmp(line, "train\n"))
        return 1;

    // Read k and n
    if (fscanf(file, "%d", &k) == EOF || fscanf(file, "%d", &n) == EOF)
    {
        fclose(file);
        return 1;
    }

    trainX = (double **)malloc(n * sizeof(double *));
    trainY = (double **)malloc(n * sizeof(double *));

    for (int i = 0; i < n; i++)
    {
        trainX[i] = (double *)malloc((k + 1) * sizeof(double));

        for (int j = 0; j <= k; j++)
        {
            if (j == 0)
            {
                trainX[i][j] = 1;
            }
            else if (fscanf(file, "%lf", &trainX[i][j]) == EOF)
            {
                fclose(file);
                return 1;
            }
        }

        trainY[i] = (double *)malloc(sizeof(double));
        if (fscanf(file, "%lf", &trainY[i][0]) == EOF)
        {
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}

int readInputFile(char *filename)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
        return 1;

    char line[256];
    fscanf(file, "%s", line);
    if (!strcmp(line, "data\n"))
        return 1;

    // Read k and m
    int attrCount;
    if (fscanf(file, "%d", &attrCount) == EOF || fscanf(file, "%d", &m) == EOF || attrCount != k)
    {
        fclose(file);
        return 1;
    }

    inputX = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++)
    {
        inputX[i] = (double *)malloc((k + 1) * sizeof(double));

        for (int j = 0; j <= k; j++)
        {
            if (j == 0)
            {
                inputX[i][j] = 1;
            }
            else if (fscanf(file, "%lf", &inputX[i][j]) == EOF)
            {
                fclose(file);
                return 1;
            }
        }
    }

    fclose(file);
    return 0;
}

double **allocateMatrix(int order)
{
    double **matrix = (double **)malloc(order * sizeof(double *));
    for (int i = 0; i < order; i++)
    {
        matrix[i] = (double *)malloc(order * sizeof(double));
    }
    return matrix;
}

void deallocateMemory(double **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

double **inverseOfMatrix(double **original, int order)
{
    // Create an augmented matrix [A|I] where 'A' is the original matrix and 'I' is the identity matrix
    double **augmentedMatrix = allocateMatrix(order);
    double **inverseMatrix = allocateMatrix(order);

    for (int i = 0; i < order; i++)
    {
        for (int j = 0; j < order; j++)
        {
            augmentedMatrix[i][j] = original[i][j];
            inverseMatrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan elimination
    for (int k = 0; k < order; k++)
    {
        for (int i = 0; i < order; i++)
        {
            if (i != k)
            {
                double factor = augmentedMatrix[i][k] / augmentedMatrix[k][k];
                for (int j = 0; j < order; j++)
                {
                    augmentedMatrix[i][j] -= factor * augmentedMatrix[k][j];
                    inverseMatrix[i][j] -= factor * inverseMatrix[k][j];
                }
            }
        }
    }

    // Divide each row by its leading coefficient to get the identity matrix on the left side
    for (int i = 0; i < order; i++)
    {
        double divisor = augmentedMatrix[i][i];
        for (int j = 0; j < order; j++)
        {
            augmentedMatrix[i][j] /= divisor;
            inverseMatrix[i][j] /= divisor;
        }
    }

    deallocateMemory(augmentedMatrix, order);
    return inverseMatrix;
}

double **transposeMatrix(double **matrix, int rows, int cols)
{
    double **transpose = (double **)malloc(cols * sizeof(double *));
    for (int i = 0; i < cols; i++)
    {
        transpose[i] = (double *)malloc(rows * sizeof(double));
        for (int j = 0; j < rows; j++)
        {
            transpose[i][j] = matrix[j][i];
        }
    }
    return transpose;
}

double **multiplyMatrices(double **matrix1, int rows1, int cols1, double **matrix2, int rows2, int cols2)
{
    if (cols1 != rows2)
    {
        printf("Matrix dimensions are not compatible for multiplication.\n");
        return NULL;
    }

    double **result = (double **)malloc(rows1 * sizeof(double *));
    for (int i = 0; i < rows1; i++)
    {
        result[i] = (double *)malloc(cols2 * sizeof(double));
    }

    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            result[i][j] = 0.0;
            for (int k = 0; k < cols1; k++)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

double **calculateWeights(double **trainX, double **trainY, int rows, int cols)
{
    // W=inverse(transpose(x)*x) * transpose(x) * y

    double **step1 = transposeMatrix(trainX, rows, cols);                     // transpose(x) : col X row
    double **step2 = multiplyMatrices(step1, cols, rows, trainX, rows, cols); // transpose(x)*x : col X col
    double **step3 = inverseOfMatrix(step2, cols);                            // inverse(transpose(x)*x) : col X col
    double **step4 = multiplyMatrices(step3, cols, cols, step1, cols, rows);  // [inverse(transpose(x)*x)] * transpose(x) : col X row
    double **step5 = multiplyMatrices(step4, cols, rows, trainY, rows, 1);    // [inverse(transpose(x)*x) * transpose(x)] * y: col X 1

    deallocateMemory(step1, cols);
    deallocateMemory(step2, cols);
    deallocateMemory(step3, cols);
    deallocateMemory(step4, cols);

    return step5;
}

int main(int argc, char *argv[])
{

    if (argc < 2 || readTrainingFile(argv[1]) || readInputFile(argv[2]))
    {
        printf("error");
        return 1;
    }

    double **W = calculateWeights(trainX, trainY, n, k + 1);
    
    double **estimatedPrices = multiplyMatrices(inputX, m, k + 1, W, k + 1, 1);
    for (int i = 0; i < m; i++)
    {
        printf("%.0f\n", estimatedPrices[i][0]);
    }

    deallocateMemory(trainX, n);
    deallocateMemory(trainY, n);
    deallocateMemory(inputX, m);
    deallocateMemory(W, k + 1);
    deallocateMemory(estimatedPrices, m);

    return 0;
}
