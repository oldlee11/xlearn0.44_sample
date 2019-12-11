//
// Created by ming.li on 19-11-29.
//

#ifndef YOUTUBEDNN_READER_DMATRIX_H_
#define YOUTUBEDNN_READER_DMATRIX_H_


#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "src/base/util.h"
#include "src/base/stl-util.h"


namespace youtubDnn {


//------------------------------------------------------------------------------
// Mapping sparse feature to dense feature. Used by distributed computation.
//------------------------------------------------------------------------------
    typedef std::unordered_map<index_t, index_t> feature_map;

//------------------------------------------------------------------------------
// Node is used to store information for each column of the feature vector.
// For tasks like LR and FM, we just need to store the feature id and the
// feature value. While for tasks like FFM, we also need to store field id.
//------------------------------------------------------------------------------
    struct Node {
        // Default constructor
        Node() { }
        Node(index_t field, index_t feat, real_t val)
                : field_id(field),
                  feat_id(feat),
                  feat_val(val) { }
        /* Field id is start from 0 */
        index_t field_id;
        /* Feature id is start from 0 */
        index_t feat_id;
        /* Feature value */
        real_t feat_val;
    };

//------------------------------------------------------------------------------
// SparseRow is used to store one line of the data, which
// is represented as a vector of the Node data structure.
//------------------------------------------------------------------------------
    typedef std::vector<Node> SparseRow;

//------------------------------------------------------------------------------
// DMatrix (data matrix) is used to store a batch of the dataset.
// It can be the whole dataset used in in-memory training, or just a
// working set used in on-disk training. This is because for many
// large-scale machine learning problems, we cannot load all the data into
// memory at once, and hence we have to load a small batch of dataset in
// DMatrix at each samplling for training or prediction.
// We can use the DMatrix like this:
//
//    DMatrix matrix;
//    for (int i = 0; i < 10; ++i) {
//      matrix.AddRow();
//      matrix.Y[i] = 0;
//      matrix.norm[i] = 1.0;
//      matrix.AddNode(i, feat_id, feat_val, field_id);
//    }
//
//    /* Serialize and Deserialize */
//    matrix.Serialize("/tmp/test.bin");
//    DMatrix new_matrix;
//    /* The new matrix is the same with old matrix */
//    new_matrix.Deserialize("/tmp/test.bin");
//
//    /* We can access the matrix */
//    for (int i = 0; i < matrix.row_length; ++i) {
//      ... matrix.Y[i] ..   /* access y */
//      SparseRow *row = matrix.row[i];
//      for (SparseRow::iterator iter = row->begin();
//           iter != row->end(); ++iter) {
//        ... iter->field_id ...   /* access field_id */
//        ... iter->feat_id ...    /* access feat_id */
//        ... iter->feat_val ...   /* access feat_val */
//      }
//    }
//
//    /* We can also get the max index of feature or field */
//    index_t max_feat = matrix.MaxFeat();
//    index_t max_field = matrix.MaxField();
//------------------------------------------------------------------------------
// TODO(aksnzhy): Implement incremental adding
    struct DMatrix {
        // Constructor
        DMatrix()
                : row_length(0),
                  row(0),
                  Y(0),
                  norm(0),
                  has_label(false),
                  pos(0) { }

        // Destructor
        ~DMatrix() { }

        // ReAlloc memoryfor the DMatrix.
        // This function will first release the original
        // memory allocated for the DMatrix, and then re-allocate
        // memory for this new matrix. For some dataset, it does not
        // contains the label y, and hence we need to set the
        // has_label variable to false. On deafult, this value will
        // be set to true.
        void ReAlloc(size_t length, bool label = true) {
            this->Reset();
            this->row_length = length;
            this->row.resize(length, nullptr);
            this->Y.resize(length, 0);
            // Here we set norm to 1.0 by default, which means
            // that we don't use instance-wise nomarlization
            this->norm.resize(length, 1.0);
            // Indicate that if current dataset has the label y
            this->has_label = label;
            this->pos = 0;
        }

        // Reset memory for DMatrix.
        void Reset() {
            this->has_label = true;
            // Delete Y
            std::vector<real_t>().swap(this->Y);
            // Delete Node
            for (int i = 0; i < this->row_length; ++i) {
                if ((this->row)[i] != nullptr) {
                    STLDeleteElementsAndClear(&(this->row));
                }
            }
            // Delete SparseRow
            std::vector<SparseRow*>().swap(this->row);
            // Delete norm
            std::vector<real_t>().swap(this->norm);
            this->row_length = 0;
            this->pos = 0;
        }

        // Dynamically adding new row for current DMatrix.
        void AddRow() {
            this->Y.push_back(0);
            this->norm.push_back(1.0);
            this->row.push_back(nullptr);
            row_length++;
        }

        // Add node to current data matrix.
        // We don't use the 'field' by default because it
        // will only be used in the ffm tasks.
        void AddNode(index_t row_id,
                     index_t feat_id,
                     real_t feat_val,
                     index_t field_id = 0) {
            // Allocate memory for the first adding
            if (row[row_id] == nullptr) {
                row[row_id] = new SparseRow;
            }
            Node node(field_id, feat_id, feat_val);
            row[row_id]->push_back(node);
        }

        // Get a mini-batch of data from curremt data matrix.
        // This method will be used for distributed computation.
        // Return the count of sample for each function call.
        index_t GetMiniBatch(index_t batch_size, DMatrix& mini_batch) {
            // Copy mini-batch
            for (index_t i = 0; i < batch_size; ++i) {
                if (this->pos >= this->row_length) {
                    return i;
                }
                mini_batch.AddRow();
                mini_batch.row[i] = this->row[pos];
                mini_batch.Y[i] = this->Y[pos];
                mini_batch.norm[i] = this->norm[pos];
                this->pos++;
            }
            return batch_size;
        }

        // We get find the max index of feature or field in current
        // data matrix. This is used for initialize our modelParameter parameter.
        inline index_t MaxFeat() const { return max_feat_or_field(true); }
        inline index_t MaxField() const { return max_feat_or_field(false); }
        inline index_t max_feat_or_field(bool is_feat) const {
            index_t max = 0;
            for (size_t i = 0; i < row_length; ++i) {
                SparseRow* sr = this->row[i];
                for (SparseRow::const_iterator iter = sr->begin();
                     iter != sr->end(); ++iter) {
                    if (is_feat) {  // feature
                        if (iter->feat_id > max) {
                            max = iter->feat_id;
                        }
                    } else {  // field
                        if (iter->field_id > max) {
                            max = iter->field_id;
                        }
                    }
                }
            }
            return max;
        }

        /* Row length of current matrix */
        index_t row_length;
        /* Store many SparseRow. Using pointer for zero-copy */
        std::vector<SparseRow*> row;
        /* (0 or -1) for negative and (+1) for positive
        examples, and others value for regression */
        std::vector<real_t> Y;
        /* Used for instance-wise normalization */
        std::vector<real_t> norm;
        /* If current dataset has label y */
        bool has_label;
        /* Current position for GetMiniBatch() */
        index_t pos;
    };

}  // namespace xLearn

#endif //YOUTUBEDNN_DMATRIX_H
