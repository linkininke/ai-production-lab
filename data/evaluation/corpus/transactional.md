Spring 的 @Transactional 只有通过代理调用才生效。同一个类内部直接调用带 @Transactional 的方法，注解不会被拦截。
