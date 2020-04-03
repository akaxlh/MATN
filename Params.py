import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
	parser.add_argument('--batch', default=32, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=8, type=int, help='embedding size')
	parser.add_argument('--memosize', default=4, type=int, help='memory size')
	parser.add_argument('--posbat', default=40, type=int, help='batch size of positive sampling')
	parser.add_argument('--negsamp', default=1, type=int, help='rate of negative sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--trn_num', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	return parser.parse_args()
args = parse_args()
# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
args.user = 19800
args.item = 22734

# swap user and item
# tem = args.user
# args.user = args.item
# args.item = tem

# args.decay_step = args.trn_num
args.decay_step = args.item//args.batch
