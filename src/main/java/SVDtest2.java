/**
 * Created by jsryu21 on 2015-05-05.
 */
import java.lang.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.solver.EigenDecomposition;

public class SVDtest2 {
    public static void main(String[] args) {
        double [][] a = new double[][] {
                {1, 2, 3, 4, 5, 6, 7},
                {3, 4, 5, 6, 7, 8, 9},
                {5, 6, 7, 8, 9, 10, 11},
                {7, 9, 11, 13, 15, 17, 19},
                {9, 11, 13, 15, 17, 19, 21},
                {11, 13, 15, 17, 19, 21, 23}
        };
        Matrix A = new DenseMatrix(a);
        int approximationCount = 3000;
        Matrix AT = A.transpose();
        Matrix ATA = AT.times(A);
        int r = Math.min(A.rowSize(), A.columnSize());
        Matrix V = A.like(A.columnSize(), A.columnSize());
        Matrix Sigma = A.like();
        for (int i = 0; i < r; ++i) {
            Vector b = ATA.viewColumn(0).like().assign(1);
            double norm = 0;
            for (int j = 0; j < approximationCount; ++j) {
                Vector tmp = ATA.times(b);
                norm = Math.sqrt(tmp.getLengthSquared());
                b = tmp.divide(norm);
            }
            V.assignColumn(i, b);
            Sigma.set(i, i, Math.sqrt(norm));
            ATA = ATA.minus(b.cross(b).times(norm));
        }
        Matrix U = A.like(A.rowSize(), A.rowSize());
        for (int i = 0; i < r; ++i) {
            U.assignColumn(i, A.times(V.viewColumn(i)).normalize());
        }
        Matrix result = U.times(Sigma).times(V.transpose());
        System.out.println(result);
    }
}
