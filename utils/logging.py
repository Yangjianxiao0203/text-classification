def log_and_write_metrics(writer, logger, results, epoch=None, verbose=True, data_type='valid'):
    # 从结果中提取指标
    acc = results["accuracy"]
    f1 = results["f1_score"]
    recall = results["recall"]
    precision = results["precision"]

    # 根据数据类型（验证或测试）来确定如何记录和写入指标
    if data_type == 'valid' and epoch is not None:
        # 将指标写入writer
        writer.add_scalar('valid_acc', acc, epoch)
        writer.add_scalar('valid_f1', f1, epoch)
        writer.add_scalar('valid_recall', recall, epoch)
        writer.add_scalar('valid_precision', precision, epoch)

        # 如果verbose为True，将指标记录到logger
        if verbose:
            logger.info(f"epoch {epoch} valid acc {acc:.4f}")
            logger.info(f"epoch {epoch} valid f1 {f1:.4f}")
            logger.info(f"epoch {epoch} valid recall {recall:.4f}")
            logger.info(f"epoch {epoch} valid precision {precision:.4f}")
    elif data_type == 'test':
        # 如果数据来自测试集，只记录和输出指标，不与特定的epoch关联
        if verbose:
            logger.info(f"test acc {acc:.4f}")
            logger.info(f"test f1 {f1:.4f}")
            logger.info(f"test recall {recall:.4f}")
            logger.info(f"test precision {precision:.4f}")
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'valid' or 'test'.")

