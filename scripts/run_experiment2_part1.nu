let haystack = [5, 5, 5, 9, 9, 9, 15, 15, 15, 21, 21, 21]
let indices = [0, 2, 4, 0, 4, 8, 0, 7, 14, 0, 10, 20]


$haystack | enumerate | each {
    |h| python src/experiment_2step.py --haystack $h.item --index ($indices | get $h.index) --multi_retain_only
}