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
        /*
        double[][] a = new double[][] {
                {3, 0, 1},
                {-1, 3, 1}
        };
        */
        Matrix A = new DenseMatrix(a);
        int numS = Math.min(6, Math.min(A.rowSize(), A.columnSize()));
        int approximationCount = 3000;

        Matrix AT = A.transpose();
        Matrix AAT = A.times(AT);
        Matrix ATA = AT.times(A);
        {
            SingularValueDecomposition asdf = new SingularValueDecomposition(A);
            Matrix result = asdf.getU().times(asdf.getS()).times(asdf.getV().transpose());
            System.out.println(result);
        }
        {
            EigenDecomposition asdf = new EigenDecomposition(AAT);
            Matrix U = asdf.getV();
            Vector sValues = asdf.getRealEigenvalues();
            asdf = new EigenDecomposition(ATA);
            Matrix VT = asdf.getV().transpose();
            for (int i = 0; i < sValues.size(); ++i) {
                if (Double.compare(sValues.get(i), 0) < 0) {
                    for (int j = i; j < sValues.size(); ++j) {
                        sValues.set(j, 0);
                    }
                    break;
                } else {
                    sValues.set(i, Math.sqrt(sValues.get(i)));
                }
            }
            Matrix sigma = A.like();
            sigma.viewDiagonal().assign(sValues);
            Matrix result = U.times(sigma).times(VT);
            System.out.println(result);
        }
        Vector sValues = A.viewDiagonal().clone().assign(0);
        Matrix U = A.like(A.rowSize(), A.rowSize());
        Matrix sigma = A.like();
        Matrix VT = A.like(A.columnSize(), A.columnSize());
        for (int i = 0; i < numS; ++i) {
            // http://mathreview.uwaterloo.ca/archive/voli/1/panju.pdf
            Vector v = AAT.viewRow(0).like().assign(1).normalize();
            for (int j = 0; j < approximationCount; ++j) {
                v = AAT.times(v).normalize();
            }
            // http://math.stackexchange.com/questions/242838/given-matrixs-eigenvectors-find-the-corresponding-sValues
            double eigenValue = 0;
            Vector AATx = AAT.times(v);
            for (int j = 0; j < AATx.size(); ++j) {
                if (Double.compare(Math.abs(AATx.get(j)), 0) != 0 && Double.compare(Math.abs(v.get(j)), 0) != 0) {
                    eigenValue = AATx.get(j) / v.get(j);
                    if (Double.compare(eigenValue, 0) > 0) {
                        break;
                    }
                }
            }
            if (Double.compare(eigenValue, 0) <= 0) {
                break;
            }
            sValues.set(i, Math.sqrt(eigenValue));
            U.assignColumn(i, v);
            Vector v2 = ATA.viewRow(0).like().assign(1).normalize();
            for (int j = 0; j < approximationCount; ++j) {
                v2 = ATA.times(v2).normalize();
            }
            VT.assignRow(i, v2);
            // http://zoro.ee.ncku.edu.tw/na2007/res/NA09.pdf
            AAT = AAT.minus(v.cross(v).times(eigenValue / v.dot(v)));
            ATA = ATA.minus(v2.cross(v2).times(eigenValue / v2.dot(v2)));
        }
        sigma.viewDiagonal().assign(sValues);
        Matrix result = U.times(sigma).times(VT);
        System.out.println(result);
    }
}
