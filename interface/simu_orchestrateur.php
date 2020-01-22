<?php

	$simu = [
		'returns_trains' => [
			'resnet' => [
				'train_acc'=> 0.85,
				'val_acc'=> 0.82
			],
			'alexnet' => [
				'train_acc'=> 0.75,
				'val_acc'=> 0.92
			],
			'knn' => [
				'train_acc'=> 0.865,
				'val_acc'=> 0.91
			]
		]
	];


	echo json_encode($simu);