# Object review of final wide EM dispatch

Only macOS15+/arm64/Eigen/non-scalar/n>=32/feature-stride1 uses the new widths.
The private bool-template driver can only be called from the four explicit
dispatch cases; all other widths/layouts retain the existing path. Fixed-size
Map borrows existing model output and source; means/variance source lifetimes
still end/start at the complete E/M phase boundary. Observation reductions
and arithmetic parenthesization remain unchanged.

V1 object common entry frame was1200B (160+1040),112B above baseline1088B.
Final V2 restores1088B (160+928); wide mean/variance driver frames224/160B.
Existing density frame is224B; the inspected own-frame numeric chain maximum
therefore stays1312B. These figures exclude system math/exception/outer frames.
The formal production object is byte-for-byte equal to the reviewed V2 object;
see object-v2-review.json for SHA, entry assembly and calls. Both wide helper
bodies and the common entry contain no FMA instructions; double-precision
NEON multiply/add instructions are present. The unchanged softmax helper does
contain explicit Eigen exponential FMA; it is outside this EM change.

No private threads, input packing/copy, numerical heap, retained state, new
operator, AST/DAG/Planner/IPC schema or Excel formula/budget is introduced.
Actual heap/RSS must still be compared through complete public requests;
unchanged planned scratch and local stack do not establish total RSS parity.
